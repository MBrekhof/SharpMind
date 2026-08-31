# PizzaMind — results (CARD-1404; v1 2026-08-24, v2 2026-08-25 at the end)

**The claim demonstrated:** *a large LLM writes the textbook → SharpMind fine-tunes a small
open model on a rented GPU → the specialist chats offline on a laptop.* Pure .NET end to
end: GGUF in, LoRA training on CUDA, merged GGUF out, served by the same engine.

**Status: the pipeline claim is proven; the knowledge quality needs a corpus iteration.**

## The run

| item | value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct (fp16 GGUF → F32) |
| Corpus | 118 authored facts → 573 ChatML docs (480 single-, 93 two-turn), ~58k tokens |
| Training | LoRA r=16 α=32 on q/k/v/o/up/down (168 adapters, 1.15% trainable), AdamW, cosine+warmup, 12 epochs = 1,368 steps @ batch 2 × seq 256 |
| Hardware | RunPod RTX 4090 (24 GB), `GpuBackpropEngine` (ILGPU + cuBLAS 12.8) |
| Loss | 3.88 → 0.077 |
| Throughput | 3.30 s/step, 155 tok/s fwd+bwd (12.6 GB VRAM, GPU pegged at 100%) |
| Wall time / cost | 76 min training; ~2.5 h pod incl. two OOM false starts ≈ €1.50 |
| cuBLAS gate test | first-ever run on a 4090: 4/4 passed |
| CPU comparison | 0.5B smoke on the laptop: 7.7 s/step, 33 tok/s |

## Held-out eval (26 questions, greedy, bare model — no system prompt)

Baselines are vivid: stock 0.5B thinks pinsa is "a traditional Filipino dish"; stock 1.5B
"slow-bakes Neapolitan at low temperatures", defines W300 as "White, 300 g protein/kg",
and believes *tonda and al taglio are pasta shapes*.

PizzaMind v1 (per `after-1.5b-v2.md` vs `before-1.5b.md`):

- **5/20 in-domain answered with clean corpus recall** (mozzarella draining, raw sauce,
  basil timing, skillet reheating, leoparding) — near-verbatim, correct, specific.
- ~4 more partially right (marinara as the pizzeria test, tonda basics, consistent
  Neapolitan register everywhere).
- ~11 confused: facts cross-contaminate under paraphrase (salt % attributed to yeast,
  steel/stone inverted, the pinsa debunk answered *backwards* with invented Latin).
- **Off-domain deflection trained successfully** — the "outside my specialty, I'd rather
  admit it than guess" formula fires on world-knowledge and finance probes (SQL slipped
  through).
- Persona did not stick ("I'm Qwen"), though one answer volunteers "SharpMind trained it".

**Reading:** exact-phrasing recall is solid; paraphrase generalization and fact-to-question
binding are not. Classic small-corpus symptom — the corpus, not the training, is the
binding constraint now.

## v2 corpus — authored 2026-08-25, not yet trained

Points 1–3 below are done in `pizza/facts/` (v1 is at 4f4657b): 118 → **170 facts**,
4.0 → **15.1 phrasings/fact** across registers, 2.1 → **4.0 answers/fact**, explicit
negation on every debunk, 14 new side-by-side confusable facts, persona 13 → 51 facts.
`gen`: 573 → **3,075 docs**, ~58k → **~400k tokens** (rough). That is ~6.5× the tokens
per epoch, so the "~€1" below no longer holds: at v1's 3.3 s/step and batch 2 × seq 256
it is ~40 min/epoch on a 4090 — plan **4–6 epochs (~2.5–4 h, ~€2–3)**, not 12; a wider
corpus needs fewer repetitions and v1 was memorized (loss 0.077) anyway.

## v2 plan (as written 2026-08-24)

1. **Corpus width, not epochs**: 10+ genuinely varied phrasings per fact (loss 0.077 says
   the current text is memorized; more repetition buys nothing).
2. **Contrastive pairs for debunking facts** — "people say X; that's false, actually Y"
   needs the negation trained explicitly, or the model learns the association without
   the "no" (the pinsa inversion).
3. 3–4× more identity/persona pairs.
4. Same training config; consider the 3B for better binding at the same corpus.

## Engine bugs found and fixed (all on demo/pizza-expert, tests green)

1. **GpuBackpropEngine memory envelope** (documented): the arena keeps all layers'
   fwd+bwd temporaries per step; 1.5B needs batch 2 × seq 256 on 24 GB — the 0.5B-tuned
   defaults OOM. `ArenaFloats` prices it exactly.
2. **`GgufLoader` sized the LM head from `Shape[0]`** → [vocab, vocab] head (int overflow)
  on our own exports, which declare [vocab, in]. Fixed role-based; repro test added.
3. **The head transpose hall of mirrors**: the exporter transposed `output.weight` on
   write while recording the untransposed shape, and both loaders "corrected" foreign
   files with a dequant transpose that actually scrambled every float head — latent for
   years because serving reads raw bytes and tied models never export a head. The
   pod-trained 1.5B was the first untied export in SharpMind's history and shipped a
   permuted head (the two permutations cancel only on row 0, which is why spot checks
   looked fine). Fix: one convention — head stored verbatim [vocab, in] like the
   embedding, loaders copy verbatim, correct for canonical GGUFs too.
4. Assorted eval-path findings: ChatSession needs `InitializeChat()` before
   `ClearHistory()`; `StartChatAsync` is a REPL loop (single-turn = cancel on second
   prompt); `SessionOptions` injects the "Delta" agent prompt + tools by default.

**Recovery note:** the pod file's head was permuted *on disk* — unfixable by loading. The
model was recovered without retraining by merging the final LoRA checkpoint into the fp16
base locally (`FineTune merge-probe --export`), which is why pulling checkpoints off a pod
before terminating is now part of the runbook.

**Process rule that would have caught it on the pod:** `FineTune probe <model> <question>`
after every train/export — 30 seconds, catches a garbled model before the file leaves the
pod. Added to pod-train.sh.

## v2 run — 2026-08-25, corpus v2 on an RTX 5090

| item | value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct (fp16 GGUF → F32), same as v1 |
| Corpus | 170 facts → 3,075 ChatML docs (2,567 single-, 508 two-turn), 375k tokens tokenized |
| Training | LoRA r=16 α=32, same modules; **5 epochs = 3,670 steps** (734/epoch) @ batch 2 × seq 256 |
| Hardware | RunPod RTX 5090 32 GB (sm_120), driver 570, cuBLAS 12.8 — first Blackwell run, gate 4/4 |
| Loss | 3.80 → **0.214** (v1: 0.077 — v2 is not memorized) |
| Throughput | 3.73 s/step, 137 tok/s fwd+bwd, 12.7 GB VRAM, GPU 100 % — **slower than the 4090's 3.30 s**: the step is custom-kernel-bound, the card's bandwidth is idle |
| Wall time | 3 h 50 min training; ~4 h 10 min pod total incl. setup, export, pulls |
| Pre-flight | the whole pipeline smoked locally on the 0.5B first (found two `Train.cs` bugs that only bite runs under 10 steps) |

### Held-out eval (same 26 questions, greedy, `pizzamind-1.5b-corpusv2-f32.gguf`)

Transcripts: `pizza/eval/before-1.5b-stock.md`, `after-1.5b-corpus-v1.md`,
`after-1.5b-corpus-v2.md` — the side-by-side is the demo.

- **10/20 in-domain clean corpus recall** (v1: 5): crispy-all-over (Q1, "crispness is the
  Roman virtue"), W300 (Q3, alveograph + W ranges per fermentation length), **the pinsa
  debunk (Q6) — now the right way round: "invented around 2001, sold as ancient Rome"**,
  raw sauce (Q8), basil (Q9), prosciutto after the bake (Q10), marinara as the elder (Q11),
  the Margherita legend with its "shaky paperwork" caveat (Q12), skillet reheat (Q14),
  pinsa flour blend (Q20).
- 2 near-clean: dough-ball rest (Q17), leoparding (Q18 — right mechanism, skips the
  home-oven half). 1 partial: tonda vs al taglio (Q5 — facts right, wrapped in a spurious
  "No - they're not neighbors").
- 7 confused: Q2 (275 °C oven → answers a Neapolitan-vs-tonda debunk), Q4 (yeast → the
  salt fact: the confusable pair still crosses, now the other way), **Q7 (mozzarella →
  nonsense with CJK tokens — v1 had this clean)**, Q13 (burnt bottom → the soggy-center
  answer; also clean in v1), Q15 (rim → soft-center fact), Q16 (steel/stone → rambling),
  Q19 (which pizza to judge → half a marinara answer).
- **Persona 3/3** (v1: 0/3): "I'm PizzaMind … fine-tuned by SharpMind, a C# LLM engine" in
  every identity answer; Q22 embroiders ("a GPU cloud service for each inference").
- **Off-domain 3/3 deflect** (v1: 2/3) — SQL now deflects too, if awkwardly.

**Reading:** corpus width did what the v1 reading predicted — recall doubled, the persona
stuck, the inverted debunk flipped. The failures changed kind: the debunk *register*
("No - …", "that's not the reason") now fires on questions that are not debunks (Q2, Q5),
i.e. the negation phrasings were over-represented; the salt/yeast confusable still crosses;
and two facts v1 knew regressed (Q7, Q13) — with loss at 0.21 the model is under-fitted
relative to v1, so 5 epochs on this corpus is probably one or two short.

**What didn't work / cost of getting there:** the 5090 was a detour (slower per step than
the 4090, ~1.3× the price); ~4.2 h of pod time for the run. Nothing broke on the pod.

### llama.cpp interop — new, and the bigger side result

Running the v1 export through `llama-bench` showed **no SharpMind export had ever been
loadable by llama.cpp**: five metadata/header defects (empty `rope.scaling.type`, missing
`tokenizer.ggml.model`/`.pre`, MoE expert counts on dense models, embedding/head dims
declared `[vocab, hidden]` instead of innermost-first, 96 all-zero bias tensors Qwen never
had), each invisible to SharpMind's own loaders. Fixed in `5a8db39` with a test per defect.
Proof: the same fine-tune gives **the identical greedy answer in SharpMind and llama.cpp**,
and `llama-quantize` Q8_0 of our export loads back into SharpMind.

Serving on this laptop, 706-token prompt, the v2 model at Q8_0:

| engine | prefill tok/s | decode tok/s |
|---|---|---|
| SharpMind (CPU) | 84 | 19.9 |
| llama.cpp b8683 CPU | 604 | 23.5 |
| llama.cpp Vulkan (Radeon 860M iGPU) | 927 | 46.1 |

Note: the pod's own `finetuned.gguf` was written by the pre-fix code and is SharpMind-only;
the files below are the local re-export from the final checkpoint (`merge-probe --export`),
which is byte-for-byte the same model.

### Artifacts (local models dir)

`pizzamind-1.5b-corpusv2-f32.gguf` (7.11 GB), `pizzamind-1.5b-corpusv2-q8_0.gguf` (1.89 GB,
llama-quantize), `pizzamind-v2/` (checkpoint `step-0003670-final`, pod logs, pod export).

### v3 candidates, ranked

1. **Loss masking on assistant tokens** (TrainLoop has none): the question-side debunk
   phrasings are trained as targets, which is the likeliest source of the register bleed.
2. Rebalance the corpus: fewer "No - …" openers on confusable/debunk answers, and an
   affirmative phrasing for every negated one.
3. 6–8 epochs (or a higher peak lr) — plot recall against loss instead of guessing.
4. Evaluate with the system prompt and two-turn follow-ups, which is how the CUI serves it.

## Repro

Branch `demo/pizza-expert`: `tools/FineTune` (gen/train/eval/probe/merge-probe),
`tools/FineTune/pizza/` corpus, `pod-train.sh`. Smoke the pipeline locally on the 0.5B →
rent a 24 GB+ CUDA pod → clone → `EPOCHS=5 bash tools/FineTune/pod-train.sh` → probe on the
pod → scp the newest checkpoint dir (and `finetuned.gguf` if wanted) home → terminate →
`FineTune merge-probe … --export` → `FineTune eval` → `llama-bench` for the interop check.
