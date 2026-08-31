using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Training;
using SharpMind.Training.LoRA;

namespace FineTune;

/// <summary>
/// Diagnostic: rebuilds the fine-tuned model in process — base GGUF + LoRA checkpoint +
/// Merge — and greedy-probes it without any export round-trip. Splits "the adapters are
/// bad" from "the export/load cycle is bad": coherent output here with garbage from the
/// exported file convicts the file cycle, and vice versa.
/// </summary>
internal static class MergeProbe
{
    public static Task<int> Run(string[] args)
    {
        if (args.Length < 3) { Console.Error.WriteLine("merge-probe <base.gguf> <checkpointDir> <question> [rank 16] [alpha rank*2] [maxNew 60]"); return Task.FromResult(1); }
        string modelPath = args[0], ckptDir = args[1], question = args[2];
        int rank = args.Length > 3 ? int.Parse(args[3]) : 16;
        float alpha = args.Length > 4 ? float.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture) : 2f * rank;
        int maxNew = args.Length > 5 ? int.Parse(args[5]) : 60;

        var fmt = ModelFormatHelpers.GetFormatForExtension(modelPath) ?? throw new InvalidDataException($"File type not supported: {modelPath}");
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
        metaHelper.Load(modelPath, null, out _, out var modelConfig, out var tokenizer);
        if (tokenizer is null) throw new InvalidOperationException("Model file has no embedded tokenizer.");
        var sharpConfig = modelConfig.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ToJigSawMapping());
        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, LoadMode.Full, quantizedResident: false);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        using var lora = new LoRAModel(model, new LoRAConfig
        {
            Rank = rank,
            Alpha = alpha,
            TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        }, seed: 42);
        var parameters = lora.LoRAParameters();
        var meta = Checkpoint.Load(ckptDir, parameters);
        Console.WriteLine($"checkpoint: step {meta.Step}, {parameters.Count} LoRA tensors loaded");

        lora.Merge();
        Console.WriteLine("merged; probing:");
        string prompt = $"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n";
        int[] ids = tokenizer.Encode(prompt, addBos: false, addEos: false);
        var outIds = SmmTrainingPipeline.GenerateGreedy(model, ids, modelConfig.VocabSize, maxNew);
        Console.WriteLine(tokenizer.Decode(outIds.Skip(ids.Length).ToArray(), skipSpecials: false));

        string? exportPath = Gen.ArgStr(args, "--export");
        if (exportPath is not null)
        {
            metaHelper.Load(modelPath, null, out var srcMeta, out _, out _);
            string smmPath = Path.ChangeExtension(exportPath, ".smm");
            SharpMind.Model.Format.SmmTrainingExporter.Export(weights, tokenizer, smmPath, chatTemplate: srcMeta.GetChatTemplate(), model: model);
            SharpMind.Model.Format.Conversion.SmmToGufConverter.Convert(smmPath, exportPath);
            File.Delete(smmPath);
            Console.WriteLine($"exported {exportPath} ({new FileInfo(exportPath).Length / 1e9:F2} GB)");
        }
        return Task.FromResult(0);
    }
}
