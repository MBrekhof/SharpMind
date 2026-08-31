using System.Text;
using System.Text.Json;

namespace FineTune;

/// <summary>
/// Expands authored facts into a shuffled ChatML train.jsonl.
///
/// Input: a directory of *.json topic files, each
///   { "facts": [ { "id": "...", "q": ["question phrasing", ...],
///                  "a": "canonical answer", "alt": ["answer variant", ...]? } ] }
///
/// Every question phrasing is paired with the canonical answer; answer variants
/// (when present) are rotated through the phrasings so the same fact is seen in
/// several wordings on both sides. A fraction of the corpus is additionally
/// assembled into two-turn conversations (two facts from the same topic file),
/// because the served model must survive follow-up questions, not just openers.
/// </summary>
internal static class Gen
{
    private sealed record Fact(string Id, string[] Q, string A, string[]? Alt);

    public static Task<int> Run(string[] args)
    {
        if (args.Length < 2) { Console.Error.WriteLine("gen <factsDir> <outDir> [--seed 42] [--multiturn 0.2]"); return Task.FromResult(1); }
        string factsDir = args[0], outDir = args[1];
        int seed = ArgInt(args, "--seed", 42);
        double multiturn = ArgDouble(args, "--multiturn", 0.2);
        Directory.CreateDirectory(outDir);

        var rng = new Random(seed);
        var docs = new List<string>();
        int factCount = 0, pairCount = 0, turnPairCount = 0;

        var files = Directory.GetFiles(factsDir, "*.json").OrderBy(f => f, StringComparer.Ordinal).ToArray();
        if (files.Length == 0) { Console.Error.WriteLine($"no *.json fact files in {factsDir}"); return Task.FromResult(1); }

        foreach (var file in files)
        {
            var doc = JsonDocument.Parse(File.ReadAllText(file));
            var facts = doc.RootElement.GetProperty("facts").EnumerateArray().Select(f => new Fact(
                f.GetProperty("id").GetString()!,
                f.GetProperty("q").EnumerateArray().Select(x => x.GetString()!).ToArray(),
                f.GetProperty("a").GetString()!,
                f.TryGetProperty("alt", out var alt) ? alt.EnumerateArray().Select(x => x.GetString()!).ToArray() : null)).ToList();
            factCount += facts.Count;

            // Single-turn: every phrasing once, answers rotating canonical + variants.
            var topicPairs = new List<(string q, string a)>();
            foreach (var fact in facts)
            {
                var answers = new List<string> { fact.A };
                if (fact.Alt is { Length: > 0 }) answers.AddRange(fact.Alt);
                for (int i = 0; i < fact.Q.Length; i++)
                    topicPairs.Add((fact.Q[i], answers[i % answers.Count]));
            }
            pairCount += topicPairs.Count;
            docs.AddRange(topicPairs.Select(p => ChatMl((p.q, p.a))));

            // Two-turn: random same-topic pairs, count scaled by --multiturn.
            int turns = (int)(topicPairs.Count * multiturn);
            for (int i = 0; i < turns; i++)
            {
                var p1 = topicPairs[rng.Next(topicPairs.Count)];
                var p2 = topicPairs[rng.Next(topicPairs.Count)];
                if (p1.q == p2.q) continue;
                docs.Add(ChatMl(p1, p2));
                turnPairCount++;
            }
        }

        // Shuffle once, seeded, so train order is reproducible.
        for (int i = docs.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (docs[i], docs[j]) = (docs[j], docs[i]);
        }

        string outPath = Path.Combine(outDir, "train.jsonl");
        using (var w = new StreamWriter(outPath, false, new UTF8Encoding(false)))
            foreach (var d in docs)
                w.WriteLine(JsonSerializer.Serialize(new { text = d }));

        long chars = docs.Sum(d => (long)d.Length);
        Console.WriteLine($"facts: {factCount} across {files.Length} topic files");
        Console.WriteLine($"docs:  {docs.Count} ({pairCount} single-turn, {turnPairCount} two-turn), ~{chars / 1e6:F1}M chars (~{chars / 4 / 1000}k tokens rough)");
        Console.WriteLine($"wrote {outPath}");
        return Task.FromResult(0);
    }

    private static string ChatMl(params (string q, string a)[] turns)
    {
        var sb = new StringBuilder();
        foreach (var (q, a) in turns)
            sb.Append("<|im_start|>user\n").Append(q).Append("<|im_end|>\n")
              .Append("<|im_start|>assistant\n").Append(a).Append("<|im_end|>\n");
        return sb.ToString();
    }

    internal static int ArgInt(string[] args, string name, int def)
        => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? int.Parse(args[i + 1]) : def;

    internal static double ArgDouble(string[] args, string name, double def)
        => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? double.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture) : def;

    internal static float ArgFloat(string[] args, string name, float def)
        => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture) : def;

    internal static string? ArgStr(string[] args, string name)
        => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}
