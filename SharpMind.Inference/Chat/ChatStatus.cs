namespace SharpMind.Inference.Chat;

/// <summary>
/// Status values during chat response generation.
/// </summary>
public enum ChatStatus
{
    /// <summary>Analyzing request, planning response.</summary>
    Thinking,
    /// <summary>Updating context/history.</summary>
    Updating,
    /// <summary>Executing tools/skills.</summary>
    Executing,
    /// <summary>Generating text response.</summary>
    Responding,
    /// <summary>Waiting for input.</summary>
    Waiting,
    /// <summary>Sub-agent is generating a response (streamed outside main history/KV cache).</summary>
    Researching,
    /// <summary>That chat was interrupted, cancelled or failed.</summary>
    Interrupted,
    /// <summary>Completed.</summary>
    Complete,
    /// <summary>
    /// The model produced a tool call and <see cref="IChatSession.ReturnToolCalls"/>
    /// is set: the parsed call is in <see cref="ChatStreamEntry.ToolCall"/>, nothing
    /// was dispatched, and the turn is over until the caller adds the result and
    /// continues with <see cref="IChatSession.GetResponseStreamAsync"/>.
    /// </summary>
    ToolCall
}
