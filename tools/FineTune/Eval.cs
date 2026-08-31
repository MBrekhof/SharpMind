using System.Diagnostics;
using System.Text;
using SharpMind.CUI.App;
using SharpMind.Inference.Chat;

namespace FineTune;

/// <summary>
/// Asks each question through the real CUI serve path (SessionLauncher -> ChatSession,
/// KV cache and all) and writes a markdown Q/A transcript. Run it once against the
/// stock GGUF and once against the fine-tuned one; the diff of the two files is the demo.
/// Deterministic: TopK=1 (greedy), fresh history per question.
/// </summary>
internal static class Eval
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2) { Console.Error.WriteLine("eval <model.gguf> <questions.txt> [--out answers.md] [--max-new 200] [--label name]"); return 1; }
        string modelPath = args[0], questionsPath = args[1];
        string outPath = Gen.ArgStr(args, "--out") ?? "answers.md";
        int maxNew = Gen.ArgInt(args, "--max-new", 200);
        string label = Gen.ArgStr(args, "--label") ?? Path.GetFileNameWithoutExtension(modelPath);

        var questions = File.ReadLines(questionsPath)
            .Select(l => l.Trim())
            .Where(l => l.Length > 0 && !l.StartsWith('#'))
            .ToList();

        var options = new SessionOptions
        {
            ModelPath = modelPath,
            Formatter = FormatterStrategy.Auto,
            Generator = GeneratorStrategy.Standard,
            // Bare model, no synthesized agent system prompt or tools — the eval measures
            // what's in the weights, not what the CUI's agent scaffolding contributes.
            SkipAgentPrompt = true,
            DisableTools = true,
        };

        var sw = Stopwatch.StartNew();
        var loadResult = await SessionLauncher.LoadModelAsync(options);
        if (!loadResult.Success) { Console.Error.WriteLine($"load failed: {loadResult.Error}"); return 1; }
        Console.WriteLine($"loaded {Path.GetFileName(modelPath)} in {sw.Elapsed.TotalSeconds:F1}s");

        var launch = SessionLauncher.BuildSession(options, loadResult.Loaded, permissions: null);
        if (!launch.Success || launch.Session is null) { Console.Error.WriteLine($"session failed: {launch.Error}"); return 1; }
        var session = launch.Session;
        session.InitializeChat();
        session.MaxNewTokens = maxNew;
        session.TopK = 1;          // greedy — reproducible transcripts
        session.Temperature = 1f;  // irrelevant at TopK=1, but explicit

        var md = new StringBuilder();
        md.AppendLine($"# {label}");
        md.AppendLine();
        md.AppendLine($"Model: `{Path.GetFileName(modelPath)}` — greedy (TopK=1), max {maxNew} new tokens, fresh history per question.");
        md.AppendLine();

        for (int i = 0; i < questions.Count; i++)
        {
            string q = questions[i];
            session.ClearHistory();
            session.ResetCaches();
            var t0 = sw.Elapsed;
            // StartChatAsync is a REPL loop: the prompt func is called again after each answer,
            // so the second call cancels to make it a single turn.
            using var cts = new CancellationTokenSource();
            bool asked = false;
            await session.StartChatAsync(() =>
            {
                if (asked) { cts.Cancel(); return ""; }
                asked = true;
                return q;
            }, _ => { }, cts.Token);
            var answer = session.History.LastOrDefault(m => m.Role == ChatRole.Agent)?.Content?.Trim() ?? "(no answer)";
            double secs = (sw.Elapsed - t0).TotalSeconds;
            Console.WriteLine($"[{i + 1}/{questions.Count}] {q}  ({secs:F1}s)");
            md.AppendLine($"**Q{i + 1}: {q}**");
            md.AppendLine();
            md.AppendLine(answer);
            md.AppendLine();
        }

        await session.DisposeAsync();
        loadResult.Loaded!.Model.Dispose();
        File.WriteAllText(outPath, md.ToString(), new UTF8Encoding(false));
        Console.WriteLine($"wrote {outPath}");
        return 0;
    }
}
