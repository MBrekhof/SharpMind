using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.AI;
using SharpMind.Inference.Chat;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using ChatRole = Microsoft.Extensions.AI.ChatRole;
using SmChat = SharpMind.Inference.Chat;

namespace SharpMind.Extensions.AI;

/// <summary>
/// <see cref="IChatClient"/> over a SharpMind <see cref="IChatSession"/>: a locally
/// loaded model behind the Microsoft.Extensions.AI abstractions, with client-side
/// tool calling via <see cref="FunctionInvokingChatClient"/>.
/// <para>
/// One session is one model with one KV cache, so calls are serialized. The
/// session's history is kept in step with the messages each call carries: when a
/// call extends what the previous one sent, only the new messages are added and
/// the session's token-prefix match keeps the KV cache; when the caller rewrote
/// history, the session is rebuilt. Tool definitions are described to the model
/// in a System message in SharpMind's own tool-call format.
/// </para>
/// <para>
/// Applied from <see cref="ChatOptions"/> (last call wins): <c>Temperature</c>,
/// <c>TopP</c>, <c>TopK</c>, <c>MaxOutputTokens</c>, <c>Tools</c>/<c>ToolMode</c>,
/// <c>Instructions</c>. Not supported: <c>StopSequences</c>, <c>Seed</c>,
/// <c>ResponseFormat</c>, <c>FrequencyPenalty</c>/<c>PresencePenalty</c>,
/// <c>ModelId</c>; <c>ChatToolMode.RequireAny</c>/<c>RequireSpecific</c> behave as Auto.
/// </para>
/// </summary>
public sealed class SharpMindChatClient(IChatSession session, bool disposeSession = false) : IChatClient
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly List<ChatMessage> _seen = [];   // the M.E.AI messages the session's history currently mirrors
    private string? _preamble;                        // tool prompt + instructions the session currently carries
    private readonly ChatClientMetadata _metadata = new("sharpmind");

    /// <summary>The session this client drives.</summary>
    public IChatSession Session => session;

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        => await GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken).ConfigureAwait(false);

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            SyncHistory(messages.ToList(), options);
            ApplyOptions(options);

            string responseId = Guid.NewGuid().ToString("N");
            string messageId = Guid.NewGuid().ToString("N");
            ChatResponseUpdate Update(AIContent content) => new(ChatRole.Assistant, [content]) { ResponseId = responseId, MessageId = messageId };

            var held = new StringBuilder();      // text not yet released, see MightBeToolCall
            var flushed = new StringBuilder();   // text the caller has received
            FunctionCallContent? call = null;
            long outputTokens = 0;
            float? tps = null, ttft = null;
            ChatFinishReason? finish = null;

            await foreach (var entry in session.GetResponseStreamAsync(null, null, cancellationToken).ConfigureAwait(false))
            {
                tps = entry.TokensPerSecond ?? tps;
                ttft = entry.TimeToFirstToken ?? ttft;
                switch (entry.Status)
                {
                    case ChatStatus.Thinking when entry.Token is { Length: > 0 }:
                        outputTokens++;
                        yield return Update(new TextReasoningContent(entry.Token));
                        break;

                    case ChatStatus.Responding when entry.Token is { Length: > 0 }:
                        outputTokens++;
                        held.Append(entry.Token);
                        if (!MightBeToolCall(held))
                        {
                            flushed.Append(held);
                            yield return Update(new TextContent(held.ToString()));
                            held.Clear();
                        }
                        break;

                    case ChatStatus.ToolCall:
                        held.Clear();
                        call = MessageMapper.ToFunctionCall(entry.ToolCall!);
                        finish = ChatFinishReason.ToolCalls;
                        yield return Update(call);
                        break;

                    case ChatStatus.Complete:
                        finish = ChatFinishReason.Stop;
                        break;

                    case ChatStatus.Interrupted:
                        if (entry.Error is not null) throw new InvalidOperationException(entry.Error);
                        cancellationToken.ThrowIfCancellationRequested();
                        finish = ChatFinishReason.Stop;
                        break;
                }
            }

            if (held.Length > 0)
            {
                flushed.Append(held);
                yield return Update(new TextContent(held.ToString()));
            }

            var usage = new UsageDetails { InputTokenCount = session.LastPrefillTokenCount, OutputTokenCount = outputTokens };
            usage.TotalTokenCount = usage.InputTokenCount + usage.OutputTokenCount;
            var last = Update(new UsageContent(usage));
            last.FinishReason = finish ?? ChatFinishReason.Stop;
            last.AdditionalProperties = new() { ["tokens_per_second"] = tps, ["time_to_first_token"] = ttft };
            yield return last;

            // The session recorded the model's turn itself; remember what the caller
            // was told so the next call's copy of it is recognised and not re-added.
            List<AIContent> produced = [];
            if (flushed.Length > 0) produced.Add(new TextContent(flushed.ToString()));
            if (call is not null) produced.Add(call);
            _seen.Add(new ChatMessage(ChatRole.Assistant, produced));
        }
        finally
        {
            _gate.Release();
        }
    }

    // ponytail: the session only knows a reply was a tool call once it is complete,
    // so text is held back while it still looks like one — a leading "{" or the
    // start of a <tool_call> tag — and released the moment it stops looking like
    // one. A genuine tool call therefore never reaches the caller as text; a plain
    // reply that happens to open with "{" arrives whole at the end of the turn.
    private static bool MightBeToolCall(StringBuilder held)
    {
        string s = held.ToString().TrimStart();
        if (s.Length == 0) return true;
        if (s[0] == '{') return true;
        const string tag = "<tool_call>";
        return s.Length < tag.Length ? tag.StartsWith(s, StringComparison.Ordinal) : s.StartsWith(tag, StringComparison.Ordinal);
    }

    private void SyncHistory(List<ChatMessage> incoming, ChatOptions? options)
    {
        session.InitializeChat();   // idempotent; ClearHistory needs the generator to exist

        string? toolPrompt = MessageMapper.BuildToolPrompt(options);
        string?[] parts = [toolPrompt, options?.Instructions];
        string? preamble = parts.Any(p => !string.IsNullOrWhiteSpace(p))
            ? string.Join("\n\n", parts.Where(p => !string.IsNullOrWhiteSpace(p)))
            : null;
        session.ReturnToolCalls = toolPrompt is not null;

        int common = 0;
        if (preamble == _preamble)
            while (common < _seen.Count && common < incoming.Count && SameMessage(_seen[common], incoming[common])) common++;

        if (common < _seen.Count || preamble != _preamble)
        {
            // The caller rewrote history, or the tools/instructions changed:
            // rebuild from scratch. ClearHistory drops the KV cache with it.
            session.ClearHistory();
            _seen.Clear();
            _preamble = preamble;
            if (preamble is not null) session.AddMessage(SmChat.ChatRole.System, preamble);
            common = 0;
        }

        foreach (var m in incoming.Skip(common))
        {
            foreach (var sm in MessageMapper.ToSharpMind(m)) session.AddMessage(sm);
            _seen.Add(m);
        }
    }

    private static bool SameMessage(ChatMessage a, ChatMessage b)
        => ReferenceEquals(a, b)
        || (a.Role == b.Role
            && a.Text == b.Text
            && ToolCalls(a).SequenceEqual(ToolCalls(b))
            && ToolResults(a).SequenceEqual(ToolResults(b)));

    private static IEnumerable<string> ToolCalls(ChatMessage m) => m.Contents.OfType<FunctionCallContent>().Select(MessageMapper.ToolCallJson);
    private static IEnumerable<string> ToolResults(ChatMessage m) => m.Contents.OfType<FunctionResultContent>().Select(r => r.CallId);

    private void ApplyOptions(ChatOptions? o)
    {
        if (o is null) return;
        if (o.Temperature is { } t) session.Temperature = t;
        if (o.TopP is { } p) session.TopP = p;
        if (o.TopK is { } k) session.TopK = k;
        if (o.MaxOutputTokens is { } n) session.MaxNewTokens = n;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        if (serviceType == typeof(ChatClientMetadata)) return _metadata;
        if (serviceType.IsInstanceOfType(session)) return session;
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose()
    {
        if (disposeSession) session.DisposeAsync().AsTask().GetAwaiter().GetResult();
        _gate.Dispose();
    }
}
