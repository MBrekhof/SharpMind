using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Training;

namespace FineTune;

/// <summary>
/// Sanity probe through the float-weights path (same load as training, no raw-byte
/// kernels, no ChatSession): greedy-decodes one ChatML question. Run it right after
/// a train/export to prove the model generates coherent text before spending time
/// (or pod money) on anything downstream. The CARD-1404 garbled-serve bug would have
/// been caught on the pod by exactly this.
/// </summary>
internal static class Probe
{
    public static Task<int> Run(string[] args)
    {
        if (args.Length < 2) { Console.Error.WriteLine("probe <model.gguf> <question> [maxNew]"); return Task.FromResult(1); }
        string modelPath = args[0], question = args[1];
        int maxNew = args.Length > 2 ? int.Parse(args[2]) : 60;

        var fmt = ModelFormatHelpers.GetFormatForExtension(modelPath) ?? throw new InvalidDataException($"File type not supported: {modelPath}");
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
        metaHelper.Load(modelPath, null, out _, out var modelConfig, out var tokenizer);
        if (tokenizer is null) throw new InvalidOperationException("Model file has no embedded tokenizer.");
        var sharpConfig = modelConfig.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ToJigSawMapping());
        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, LoadMode.Full, quantizedResident: false);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        string prompt = $"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n";
        int[] ids = tokenizer.Encode(prompt, addBos: false, addEos: false);
        string back = tokenizer.Decode(ids, skipSpecials: false);
        Console.WriteLine($"tokenizer: vocab={tokenizer.VocabSize}, prompt {ids.Length} ids [{string.Join(",", ids.Take(12))}…], round-trip ok={back == prompt}");
        var outIds = SmmTrainingPipeline.GenerateGreedy(model, ids, modelConfig.VocabSize, maxNew);
        Console.WriteLine(tokenizer.Decode(outIds.Skip(ids.Length).ToArray(), skipSpecials: false));
        return Task.FromResult(0);
    }
}
