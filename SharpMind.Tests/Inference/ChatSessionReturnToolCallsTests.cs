using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// A host that owns the tool loop (an IChatClient adapter) needs the session to
/// hand a tool call back instead of dispatching it, and to continue the turn once
/// the host has added the result — without inventing a User message to do so.
/// </summary>
public sealed class ChatSessionReturnToolCallsTests
{
    private static async Task<List<ChatStreamEntry>> Drain(IAsyncEnumerable<ChatStreamEntry> stream)
    {
        var entries = new List<ChatStreamEntry>();
        await foreach (var e in stream) entries.Add(e);
        return entries;
    }

    [Fact]
    public async Task ReturnToolCalls_HandsTheCallBack_AndDoesNotDispatch()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add",|"arguments":{"a":1,"b":2}}""");
        session.ReturnToolCalls = true;

        var entries = await Drain(session.GetResponseStreamAsync("1+2?"));

        var last = entries[^1];
        Assert.Equal(ChatStatus.ToolCall, last.Status);
        Assert.True(last.IsComplete);
        Assert.Equal("add", last.ToolCall!["tool"]!.GetValue<string>());
        Assert.Equal(2, last.ToolCall["arguments"]!["b"]!.GetValue<int>());
        // The call is recorded as the model's own turn; nothing was dispatched.
        Assert.Equal(ChatRole.Agent, session.History[^1].Role);
        Assert.DoesNotContain("Tool result:", session.GetFormattedPrompt());
    }

    [Fact]
    public async Task NullUserInput_ContinuesTheTurn_WithoutAddingAUserMessage()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add","arguments":{"a":1,"b":2}}""", "3");
        session.ReturnToolCalls = true;
        await Drain(session.GetResponseStreamAsync("1+2?"));

        session.AddMessage(ChatRole.System, "Tool result: 3");
        var entries = await Drain(session.GetResponseStreamAsync(null));

        Assert.Equal(ChatStatus.Complete, entries[^1].Status);
        Assert.Equal("3", string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token)));
        Assert.Equal(1, session.History.Count(m => m.Role == ChatRole.User));
        Assert.Equal("3", session.History[^1].Content);
        Assert.Contains("Tool result: 3", session.GetFormattedPrompt());
    }

    [Fact]
    public async Task NameIsAcceptedAsAnAliasForTool()
    {
        await using var session = ScriptedSession.Create("""<tool_call>|{"name":"add","arguments":{"a":1}}|</tool_call>""");
        session.ReturnToolCalls = true;

        var last = (await Drain(session.GetResponseStreamAsync("go")))[^1];

        Assert.Equal(ChatStatus.ToolCall, last.Status);
        Assert.Equal("add", last.ToolCall!["tool"]!.GetValue<string>());
    }

    [Fact]
    public async Task WithoutReturnToolCalls_ToolJsonIsJustAReply()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add","arguments":{"a":1}}""");

        var last = (await Drain(session.GetResponseStreamAsync("go")))[^1];

        Assert.Equal(ChatStatus.Complete, last.Status);
        Assert.Null(last.ToolCall);
    }
}
