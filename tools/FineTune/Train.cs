using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;
using SharpMind.GPU;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using SharpMind.Training;
using SharpMind.Training.LoRA;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace FineTune;

/// <summary>
/// GGUF -> LoRA fine-tune -> merge -> export .gguf. The proven CARD-1348 round-trip
/// (same load path the CUI's TrainRunner uses), with the GPU engine selected the same
/// way GpuTrainBench does it.
/// </summary>
internal static class Train
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2) { Console.Error.WriteLine("train <model.gguf> <train.jsonl> [--gpu] [--rank 16] [--alpha rank*2] [--epochs 3] [--batch 8] [--seq 512] [--lr 1e-4] [--out out] [--ckpt-interval 200] [--resume <ckpt>]"); return 1; }
        string modelPath = args[0], dataPath = args[1];
        bool useGpu = args.Contains("--gpu");
        int rank = Gen.ArgInt(args, "--rank", 16);
        float alpha = Gen.ArgFloat(args, "--alpha", 2f * rank);
        int epochs = Gen.ArgInt(args, "--epochs", 3);
        int batch = Gen.ArgInt(args, "--batch", 8);
        int seq = Gen.ArgInt(args, "--seq", 512);
        float lr = Gen.ArgFloat(args, "--lr", 1e-4f);
        string outDir = Gen.ArgStr(args, "--out") ?? "out";
        int ckptInterval = Gen.ArgInt(args, "--ckpt-interval", 200);
        string? resume = Gen.ArgStr(args, "--resume");
        Directory.CreateDirectory(outDir);

        var sw = Stopwatch.StartNew();
        void Stage(string s) => Console.WriteLine($"[{sw.Elapsed.TotalSeconds,7:F1}s] {s}");

        // ── load GGUF as F32 training transformer (CUI's TrainRunner path) ──
        var fmt = ModelFormatHelpers.GetFormatForExtension(modelPath) ?? throw new InvalidDataException($"File type not supported: {modelPath}");
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
        metaHelper.Load(modelPath, null, out var meta, out var modelConfig, out var tokenizer);
        if (tokenizer is null) throw new InvalidOperationException("Model file has no embedded tokenizer.");
        var sharpConfig = modelConfig.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ToJigSawMapping());
        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, LoadMode.Full, quantizedResident: false);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        Stage($"model: {Path.GetFileName(modelPath)} {modelConfig.HiddenDim}h {modelConfig.NumLayers}L vocab={modelConfig.VocabSize}");

        using var lora = new LoRAModel(model, new LoRAConfig
        {
            Rank = rank,
            Alpha = alpha,
            TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        }, seed: 42);
        var parameters = lora.LoRAParameters();
        Stage($"LoRA r={rank} alpha={alpha}: {lora.AdapterCount} adapters, {lora.TrainableRatio():P2} trainable");

        // ── dataset: tokenize once for exact step math, then stream repeats ──
        var docs = File.ReadLines(dataPath)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .Select(l => JsonDocument.Parse(l).RootElement.GetProperty("text").GetString()!)
            .ToList();
        long totalTokens = docs.Sum(d => (long)tokenizer.Encode(d, addBos: false, addEos: false).Length);
        int stepsPerEpoch = (int)Math.Ceiling(totalTokens / (double)(batch * seq));
        int totalSteps = stepsPerEpoch * epochs;
        Stage($"data: {docs.Count} docs, {totalTokens / 1000.0:F0}k tokens -> {stepsPerEpoch} steps/epoch x {epochs} epochs = {totalSteps} steps");

        int eos = tokenizer.EosId >= 0 ? tokenizer.EosId : 0;
        int pad = tokenizer.PadId >= 0 ? tokenizer.PadId : eos;
        var source = new RepeatingListSource(docs, repeat: epochs + 1);
        var pipeline = PipelineNode.From(source);
        var batcher = new PackingBatcher(batchSize: batch, maxSeqLen: seq, eosTokenId: eos, padTokenId: pad);
        var loader = new DataLoader(pipeline, s => tokenizer.Encode(s, addBos: false, addEos: false), batcher, prefetchBuffer: 4);

        // ── engine + loop ──
        var ops = TrainingOpsFactory.Create(sharpConfig);
        using var optimizer = new AdamW(parameters, ops, lr: lr, weightDecay: 0f);
        int warmup = Math.Min(Math.Max(10, totalSteps / 20), totalSteps); // smoke runs can be < 10 steps
        var scheduler = new CosineWithWarmup(maxLr: lr, minLr: lr / 10f, warmupSteps: warmup, decaySteps: totalSteps);

        ITrainingEngine? engine = null;
        try
        {
            if (useGpu)
            {
                GpuBackpropEngine.ValidateSupported(model, parameters, sharpConfig);
                var device = GpuDevice.Shared;
                Stage($"GPU device: {device.Description}");
                engine = new GpuBackpropEngine(device, model, parameters, sharpConfig, batch, seq);
            }

            var cfg = new TrainConfig
            {
                TotalSteps = totalSteps,
                GradAccumSteps = 1,
                GradClipNorm = 1f,
                LogInterval = 1, // TrainLoop only calls onStep on LogInterval multiples; the callback below does its own filtering
                CheckpointInterval = ckptInterval,
                CheckpointDir = Path.Combine(outDir, "checkpoints"),
                KeepRecent = 2,
                ResumeFrom = resume,
            };
            var loop = new TrainLoop(model, parameters, loader, optimizer, scheduler, ops, smmConfig: sharpConfig, config: cfg, engine: engine);

            float firstLoss = -1, lastLoss = -1;
            var stepTimes = new List<double>();
            await loop.RunAsync(onStep: r =>
            {
                if (firstLoss < 0) firstLoss = r.Loss;
                lastLoss = r.Loss;
                stepTimes.Add(r.StepTime.TotalSeconds);
                if (r.Step <= 3 || r.Step % 10 == 0 || r.Step == totalSteps)
                {
                    double steady = stepTimes.Count > 2 ? stepTimes.Skip(2).Average() : stepTimes.Average();
                    var eta = TimeSpan.FromSeconds(steady * (totalSteps - r.Step));
                    Console.WriteLine($"  step {r.Step,5}/{totalSteps}  loss {r.Loss:F4}  gn {r.GradNorm:F2}  {r.StepTime.TotalSeconds,6:F2}s  {batch * seq / r.StepTime.TotalSeconds,6:F0} tok/s  eta {eta:hh\\:mm\\:ss}");
                }
            });
            double perStep = stepTimes.Count > 2 ? stepTimes.Skip(2).Average() : stepTimes.Average();
            Stage($"trained: loss {firstLoss:F4} -> {lastLoss:F4}, {perStep:F2}s/step, {batch * seq / perStep:F0} tok/s fwd+bwd");
        }
        finally
        {
            engine?.Dispose();
        }

        // ── merge + export ──
        lora.Merge();
        Stage("LoRA merged into base weights");
        string smmPath = Path.Combine(outDir, "finetuned.smm");
        string ggufPath = Path.Combine(outDir, "finetuned.gguf");
        SmmTrainingExporter.Export(weights, tokenizer, smmPath, chatTemplate: meta.GetChatTemplate(), model: model);
        SmmToGufConverter.Convert(smmPath, ggufPath);
        File.Delete(smmPath);
        Stage($"exported {ggufPath} ({new FileInfo(ggufPath).Length / 1e9:F2} GB)");
        return 0;
    }

    private sealed class RepeatingListSource(IReadOnlyList<string> docs, int repeat) : IDataSource
    {
        public long? EstimatedCount => (long)docs.Count * repeat;
        public string Description => $"in-memory {docs.Count} docs x{repeat}";
        public async IAsyncEnumerable<string> ReadAsync([EnumeratorCancellation] CancellationToken ct = default)
        {
            for (int r = 0; r < repeat; r++)
                foreach (var d in docs) { ct.ThrowIfCancellationRequested(); yield return d; }
            await Task.CompletedTask;
        }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
