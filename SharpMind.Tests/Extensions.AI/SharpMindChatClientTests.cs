using Microsoft.Extensions.AI;
using SharpMind.Extensions.AI;
using SharpMind.Tests.Inference;
using Xunit;

namespace SharpMind.Tests.Extensions.AI;

/// <summary>
/// The adapter over the real ChatSession with scripted model output: messages
/// land in the session, replies stream back as M.E.AI updates, and history is
/// kept in step across calls so the KV cache can do its job.
/// </summary>
public sealed class SharpMindChatClientTests
{
    private static async Task<List<ChatResponseUpdate>> Stream(IChatClient client, IList<ChatMessage> messages, ChatOptions? options = null)
    {
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync(messages, options)) updates.Add(u);
        return updates;
    }

    [Fact]
    public async Task MapsMessagesIntoTheSession_AndStreamsThinkingAndText()
    {
        await using var session = ScriptedSession.Create("<think>|plan|</think>|Hel|lo");
        using var client = new SharpMindChatClient(session);

        var updates = await Stream(client, [new(ChatRole.System, "Be terse."), new(ChatRole.User, "hi")]);

        Assert.Equal("plan", string.Concat(updates.SelectMany(u => u.Contents.OfType<TextReasoningContent>()).Select(c => c.Text)));
        Assert.Equal("Hello", string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(ChatFinishReason.Stop, updates[^1].FinishReason);
        Assert.Contains("Be terse.", session.GetFormattedPrompt());
        // History holds the user turn and the model's own (raw) reply; System messages are not listed.
        Assert.Equal(["hi", "<think>plan</think>Hello"], session.History.Select(m => m.Content));
    }

    [Fact]
    public async Task SecondCall_AppendsOnlyTheNewMessages()
    {
        await using var session = ScriptedSession.Create("one", "two");
        using var client = new SharpMindChatClient(session);
        List<ChatMessage> history = [new(ChatRole.User, "first")];

        var r1 = await client.GetResponseAsync(history);
        history.AddRange(r1.Messages);
        history.Add(new(ChatRole.User, "second"));
        var r2 = await client.GetResponseAsync(history);

        Assert.Equal("one", r1.Text);
        Assert.Equal("two", r2.Text);
        Assert.Equal(["first", "one", "second", "two"], session.History.Select(m => m.Content));
    }

    [Fact]
    public async Task RewrittenHistory_RebuildsTheSession()
    {
        await using var session = ScriptedSession.Create("one", "two");
        using var client = new SharpMindChatClient(session);

        await client.GetResponseAsync([new(ChatRole.User, "first")]);
        var r2 = await client.GetResponseAsync([new(ChatRole.User, "different")]);

        Assert.Equal("two", r2.Text);
        Assert.Equal(["different", "two"], session.History.Select(m => m.Content));
    }

    [Fact]
    public async Task AReplyOpeningWithABrace_IsDeliveredWholeAtTheEnd()
    {
        await using var session = ScriptedSession.Create("""{"an|swer"|: 42}""");   // not a tool call: no "tool"
        using var client = new SharpMindChatClient(session);

        var updates = await Stream(client, [new(ChatRole.User, "json please")]);

        var withText = updates.Where(u => u.Text.Length > 0).ToList();
        Assert.Single(withText);
        Assert.Equal("""{"answer": 42}""", withText[0].Text);
    }

    [Fact]
    public async Task AToolCall_NeverLeaksAsText_AndEndsTheTurnWithToolCalls()
    {
        await using var session = ScriptedSession.Create("""{"tool":"add",|"arguments":{"a":1,"b":2}}""");
        using var client = new SharpMindChatClient(session);
        var options = new ChatOptions { Tools = [AIFunctionFactory.Create((int a, int b) => a + b, "add")] };

        var updates = await Stream(client, [new(ChatRole.User, "1+2?")], options);

        Assert.All(updates, u => Assert.Equal("", u.Text));
        var call = Assert.Single(updates.SelectMany(u => u.Contents.OfType<FunctionCallContent>()));
        Assert.Equal("add", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, updates[^1].FinishReason);
        Assert.Contains("## Available Tools", session.GetFormattedPrompt());
    }

    [Fact]
    public async Task OptionsAndUsage_AreApplied_AndReported()
    {
        await using var session = ScriptedSession.Create("ok");
        using var client = new SharpMindChatClient(session);

        var r = await client.GetResponseAsync([new(ChatRole.User, "hi")], new ChatOptions { Temperature = 0.3f, TopK = 7, MaxOutputTokens = 99 });

        Assert.Equal(0.3f, session.Temperature);
        Assert.Equal(7, session.TopK);
        Assert.Equal(99, session.MaxNewTokens);
        Assert.Equal(1, r.Usage!.OutputTokenCount);        // "ok" is one scripted fragment
        Assert.True(r.Usage.InputTokenCount > 0);
        Assert.Same(session, client.GetService(typeof(SharpMind.Inference.Chat.IChatSession)));
        Assert.Equal("sharpmind", client.GetService<ChatClientMetadata>()!.ProviderName);
    }
}
