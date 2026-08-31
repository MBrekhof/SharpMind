// CARD-1404: the "fine-tune from GGUF" entry point the CARD-1348 findings called the
// missing piece, as a standalone CLI. Three subcommands covering the whole demo pipeline:
//
//   gen   <factsDir> <outDir>                 facts JSON -> ChatML train.jsonl (+ stats)
//   train <model.gguf> <train.jsonl> [opts]   LoRA fine-tune -> merge -> export .gguf
//   eval  <model.gguf> <questions.txt> [opts] ask each question through the real CUI serve path
//
// NOT in SharpMind.sln — build/run directly: dotnet run -c Release --project tools/FineTune -- <cmd> ...

return args.Length == 0
    ? Usage()
    : args[0].ToLowerInvariant() switch
    {
        "gen" => await FineTune.Gen.Run(args[1..]),
        "train" => await FineTune.Train.Run(args[1..]),
        "eval" => await FineTune.Eval.Run(args[1..]),
        "probe" => await FineTune.Probe.Run(args[1..]),
        "merge-probe" => await FineTune.MergeProbe.Run(args[1..]),
        _ => Usage(),
    };

static int Usage()
{
    Console.WriteLine("""
        usage:
          FineTune gen   <factsDir> <outDir> [--seed 42] [--multiturn 0.2]
          FineTune train <model.gguf> <train.jsonl> [--gpu] [--rank 16] [--alpha R*2] [--epochs 3]
                         [--batch 8] [--seq 512] [--lr 1e-4] [--out out] [--ckpt-interval 200] [--resume <ckpt>]
          FineTune eval  <model.gguf> <questions.txt> [--out answers.md] [--max-new 200] [--label name]
        """);
    return 1;
}
