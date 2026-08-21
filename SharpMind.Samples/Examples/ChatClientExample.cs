using Microsoft.Extensions.AI;
using SharpMind.Core.Quantization;
using SharpMind.Extensions.AI;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using ChatRole = Microsoft.Extensions.AI.ChatRole;

namespace SharpMind.Samples.Examples;

/// <summary>
/// A local GGUF model behind Microsoft.Extensions.AI's IChatClient, with a tool
/// the model can call. FunctionInvokingChatClient runs the tool and feeds the
/// result back; SharpMindChatClient keeps the session (and its KV cache) in step.
/// </summary>
public static class ChatClientExample
{
    public static async Task RunAsync(string modelPath)
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
        if (tokenizer is null) { Console.WriteLine("No tokenizer data in this file."); return; }

        var sharpConfig = modelConfig.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, LoadMode.Full);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig);

        await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
        {
            MaxTokens = 2048,
            Temperature = 0f,
            TopK = 1,
        };

        using IChatClient client = new FunctionInvokingChatClient(new SharpMindChatClient(session));

        var options = new ChatOptions
        {
            Tools = [AIFunctionFactory.Create(GetWeather, "get_weather", "Current weather for a city.")],
        };

        List<ChatMessage> history = [new(ChatRole.User, "What's the weather in Delft right now?")];
        await foreach (var update in client.GetStreamingResponseAsync(history, options))
            Console.Write(update.Text);
        Console.WriteLine();
    }

    private static string GetWeather(string city) => $"{city}: 19°C, light rain";
}
