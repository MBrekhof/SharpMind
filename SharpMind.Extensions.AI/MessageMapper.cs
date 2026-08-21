using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using SharpMind.Inference.Agent;
using SmChat = SharpMind.Inference.Chat;

namespace SharpMind.Extensions.AI;

/// <summary>Translates between Microsoft.Extensions.AI messages and SharpMind's chat history.</summary>
internal static class MessageMapper
{
    /// <summary>
    /// System prompt that teaches the model SharpMind's tool-call shape for the
    /// functions in <paramref name="options"/>, or null when there are none (or
    /// <see cref="ChatToolMode.None"/>). Mirrors the tool sections of
    /// <see cref="AgentBuilder.BuildAgentPrompt"/> minus its "answer only in
    /// JSON" rules: an IChatClient host wants plain text when no tool is needed.
    /// </summary>
    public static string? BuildToolPrompt(ChatOptions? options)
    {
        if (options?.Tools is not { Count: > 0 } tools || options.ToolMode is NoneChatToolMode) return null;

        var defs = new JsonArray();
        foreach (var f in tools.OfType<AIFunctionDeclaration>())
        {
            defs.Add(new JsonObject
            {
                ["name"] = f.Name,
                ["description"] = f.Description,
                ["parameters"] = JsonNode.Parse(f.JsonSchema.GetRawText()),
            });
        }
        if (defs.Count == 0) return null;

        return $$"""
            ## Tool Call Format
            To use a tool, respond ONLY with this JSON and nothing else: { "tool": "<name>", "arguments": { ... } }
            - Call one tool at a time. Wait for the result before proceeding.
            - Never invent tool names or argument values.
            - When no tool is needed, answer in plain text.

            ## Available Tools
            {{AgentBuilder.BuildCompactToolList(defs)}}
            """;
    }

    /// <summary>
    /// SharpMind messages for one M.E.AI message. A Tool message with N results
    /// becomes N <c>Tool result:</c> System lines — the shape ChatSession's own
    /// loop feeds back, so the model sees the same thing either way.
    /// </summary>
    public static IEnumerable<SmChat.ChatMessage> ToSharpMind(ChatMessage message)
    {
        if (message.Role == ChatRole.Tool)
        {
            foreach (var r in message.Contents.OfType<FunctionResultContent>())
                yield return SmChat.ChatMessage.System("Tool result: " + ResultText(r.Result));
            yield break;
        }

        if (message.Role == ChatRole.Assistant && message.Contents.OfType<FunctionCallContent>().FirstOrDefault() is { } call)
        {
            yield return SmChat.ChatMessage.Agent(ToolCallJson(call));
            yield break;
        }

        var role = message.Role == ChatRole.System ? SmChat.ChatRole.System
                 : message.Role == ChatRole.Assistant ? SmChat.ChatRole.Agent
                 : SmChat.ChatRole.User;
        yield return new SmChat.ChatMessage { Role = role, Content = message.Text, Name = message.AuthorName };
    }

    /// <summary>The JSON the model would have written for this call.</summary>
    public static string ToolCallJson(FunctionCallContent call)
    {
        var args = call.Arguments is null ? new JsonObject()
                 : JsonSerializer.SerializeToNode(call.Arguments, AIJsonUtilities.DefaultOptions)?.AsObject() ?? new JsonObject();
        return new JsonObject { ["tool"] = call.Name, ["arguments"] = args }.ToJsonString();
    }

    /// <summary>
    /// The model's tool call as M.E.AI content. Arguments come through as
    /// <see cref="JsonElement"/>s, which <see cref="AIFunction.InvokeAsync"/> binds
    /// to the function's parameter types itself.
    /// </summary>
    public static FunctionCallContent ToFunctionCall(JsonObject toolCall)
    {
        Dictionary<string, object?>? args = null;
        if (toolCall["arguments"] is JsonObject a)
        {
            args = [];
            foreach (var (name, value) in a)
                args[name] = value is null ? null : JsonSerializer.Deserialize<JsonElement>(value);
        }
        return new FunctionCallContent(Guid.NewGuid().ToString("N"), toolCall["tool"]!.GetValue<string>(), args);
    }

    private static string ResultText(object? result) => result switch
    {
        null => "null",
        string s => s,
        JsonElement { ValueKind: JsonValueKind.String } e => e.GetString()!,
        _ => JsonSerializer.Serialize(result, AIJsonUtilities.DefaultOptions),
    };
}
