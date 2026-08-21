using Microsoft.Extensions.AI;
using SharpMind.Extensions.AI;
using SharpMind.Tests.Inference;
using Xunit;
using SmChat = SharpMind.Inference.Chat;

namespace SharpMind.Tests.Extensions.AI;

/// <summary>
/// The whole point of the adapter: M.E.AI's own FunctionInvokingChatClient runs
/// the function the model asked for and feeds the result back, and the session
/// continues the same turn with that result in its history.
/// </summary>
public sealed class FunctionInvokingRoundTripTests
{
    private static readonly ChatOptions WithAdd = new() { Tools = [AIFunctionFactory.Create((int a, int b) => a + b, "add", "Adds two integers.")] };

    [Fact]
    public async Task NonStreaming_ToolCallIsInvoked_AndTheModelAnswersWithTheResult()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add","arguments":{"a":1,"b":2}}""", "3");
        using IChatClient client = new FunctionInvokingChatClient(new SharpMindChatClient(session));

        var response = await client.GetResponseAsync([new(ChatRole.User, "1+2?")], WithAdd);

        Assert.Equal("3", response.Text);
        Assert.Contains("Tool result: 3", session.GetFormattedPrompt());
        // One user turn; the session continued after the result instead of starting a new turn.
        Assert.Equal(1, session.History.Count(m => m.Role == SmChat.ChatRole.User));
        Assert.Contains(response.Messages, m => m.Contents.OfType<FunctionCallContent>().Any());
        Assert.Contains(response.Messages, m => m.Contents.OfType<FunctionResultContent>().Any());
    }

    [Fact]
    public async Task Streaming_ToolCallIsInvoked_AndOnlyTheFinalAnswerIsText()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add",|"arguments":{"a":1,"b":2}}""", "3");
        using IChatClient client = new FunctionInvokingChatClient(new SharpMindChatClient(session));

        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync([new(ChatRole.User, "1+2?")], WithAdd)) updates.Add(u);

        Assert.Equal("3", string.Concat(updates.Select(u => u.Text)));
        Assert.Contains("Tool result: 3", session.GetFormattedPrompt());
    }

    [Fact]
    public async Task ANextUserTurn_AfterAToolCall_AppendsWithoutRebuilding()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add","arguments":{"a":1,"b":2}}""", "3", "yes");
        using IChatClient client = new FunctionInvokingChatClient(new SharpMindChatClient(session));
        List<ChatMessage> history = [new(ChatRole.User, "1+2?")];

        var r1 = await client.GetResponseAsync(history, WithAdd);
        history.AddRange(r1.Messages);
        history.Add(new(ChatRole.User, "sure?"));
        var r2 = await client.GetResponseAsync(history, WithAdd);

        Assert.Equal("yes", r2.Text);
        Assert.Equal(["1+2?", """{"tool":"add","arguments":{"a":1,"b":2}}""", "3", "sure?", "yes"], session.History.Select(m => m.Content));
    }
}
