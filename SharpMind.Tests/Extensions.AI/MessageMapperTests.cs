using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using SharpMind.Extensions.AI;
using Xunit;
using SmChat = SharpMind.Inference.Chat;

namespace SharpMind.Tests.Extensions.AI;

public sealed class MessageMapperTests
{
    private static readonly AIFunction Add = AIFunctionFactory.Create((int a, int b) => a + b, "add", "Adds two integers.");

    [Fact]
    public void SystemUserAssistant_MapToTheirSharpMindRoles()
    {
        Assert.Equal((SmChat.ChatRole.System, "be terse"), RoleAndContent(new ChatMessage(ChatRole.System, "be terse")));
        Assert.Equal((SmChat.ChatRole.User, "hi"), RoleAndContent(new ChatMessage(ChatRole.User, "hi")));
        Assert.Equal((SmChat.ChatRole.Agent, "hello"), RoleAndContent(new ChatMessage(ChatRole.Assistant, "hello")));
    }

    [Fact]
    public void AssistantToolCall_BecomesTheModelsToolJson()
    {
        var msg = new ChatMessage(ChatRole.Assistant, [new FunctionCallContent("c1", "add", new Dictionary<string, object?> { ["a"] = 1, ["b"] = 2 })]);

        var (role, content) = RoleAndContent(msg);

        Assert.Equal(SmChat.ChatRole.Agent, role);
        var json = JsonNode.Parse(content)!.AsObject();
        Assert.Equal("add", json["tool"]!.GetValue<string>());
        Assert.Equal(2, json["arguments"]!["b"]!.GetValue<int>());
    }

    [Fact]
    public void ToolResults_BecomeSystemToolResultLines_OnePerResult()
    {
        var msg = new ChatMessage(ChatRole.Tool, [new FunctionResultContent("c1", 3), new FunctionResultContent("c2", "sunny")]);

        var mapped = MessageMapper.ToSharpMind(msg).ToList();

        Assert.Equal(2, mapped.Count);
        Assert.All(mapped, m => Assert.Equal(SmChat.ChatRole.System, m.Role));
        Assert.Equal("Tool result: 3", mapped[0].Content);
        Assert.Equal("Tool result: sunny", mapped[1].Content);
    }

    [Fact]
    public void ToolPrompt_DescribesTheFunctions_InSharpMindsFormat()
    {
        var prompt = MessageMapper.BuildToolPrompt(new ChatOptions { Tools = [Add] });

        Assert.NotNull(prompt);
        Assert.Contains("## Tool Call Format", prompt);
        Assert.Contains("""{ "tool": "<name>", "arguments": { ... } }""", prompt);
        Assert.Contains("## Available Tools", prompt);
        Assert.Contains("\"add\"", prompt);
        Assert.Contains("Adds two integers.", prompt);
    }

    [Fact]
    public void ToolPrompt_IsNull_WithoutToolsOrWhenToolModeIsNone()
    {
        Assert.Null(MessageMapper.BuildToolPrompt(null));
        Assert.Null(MessageMapper.BuildToolPrompt(new ChatOptions()));
        Assert.Null(MessageMapper.BuildToolPrompt(new ChatOptions { Tools = [Add], ToolMode = ChatToolMode.None }));
    }

    [Fact]
    public void ModelToolCall_BecomesFunctionCallContent_WithJsonElementArguments()
    {
        var call = MessageMapper.ToFunctionCall(JsonNode.Parse("""{"tool":"add","arguments":{"a":1,"b":2}}""")!.AsObject());

        Assert.Equal("add", call.Name);
        Assert.False(string.IsNullOrEmpty(call.CallId));
        Assert.Equal(JsonValueKind.Number, Assert.IsType<JsonElement>(call.Arguments!["b"]).ValueKind);
    }

    private static (SmChat.ChatRole, string) RoleAndContent(ChatMessage m)
    {
        var one = Assert.Single(MessageMapper.ToSharpMind(m));
        return (one.Role, one.Content);
    }
}
