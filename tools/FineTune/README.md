# FineTune — GGUF in, subject expert out

The "fine-tune from GGUF" entry point as a standalone CLI (CARD-1404), built on the
round-trip the CARD-1348 spike proved: `ModelFactory` F32 load → `LoRAModel` →
`TrainLoop` (CPU `BackpropEngine` or `SharpMind.GPU.GpuBackpropEngine`) →
`Merge()` → `SmmTrainingExporter` → `SmmToGufConverter` → a servable `.gguf`.

Not in `SharpMind.sln` — run directly:

```bash
# 1. facts JSON -> ChatML train.jsonl
dotnet run -c Release --project tools/FineTune -- gen tools/FineTune/pizza/facts out-pizza

# 2. LoRA fine-tune (drop --gpu for the CPU engine)
dotnet run -c Release --project tools/FineTune -- train model.gguf out-pizza/train.jsonl \
    --gpu --rank 16 --epochs 6 --batch 8 --seq 512 --lr 1e-4 --out out-pizza

# 3. ask questions through the real CUI serve path (greedy, fresh history per question)
dotnet run -c Release --project tools/FineTune -- eval out-pizza/finetuned.gguf \
    tools/FineTune/pizza/eval-questions.txt --out after.md --label PizzaMind
```

Run `eval` against the stock GGUF and the fine-tuned one; the diff of the two
markdown files is the demo.

## The pizza corpus

`pizza/facts/*.json` is an LLM-authored knowledge base on European thin-crust
pizza: 170 facts (v2, 2026-08-25) across styles, dough, baking, toppings,
history, troubleshooting, and the PizzaMind persona (identity + honest
off-domain deflection). Each fact carries ~15 question phrasings spanning
registers (direct, casual, first-person scenario, third-party claim,
misconception-led, contrast, terse) and 4 answer variants; debunk and
confusable facts name the wrong claim and reject it explicitly, and confusable
pairs get a dedicated side-by-side fact (`salt-vs-yeast-roles`,
`steel-or-stone-decision`, `tonda-vs-al-taglio`, ...). `gen` expands them into
single- and two-turn ChatML docs (~3,075 docs, ~400k tokens rough). v1 (118
facts, 573 docs, ~58k tokens — see `RESULTS-pizzamind-v1.md`) is at 4f4657b.
`pizza/eval-questions.txt` holds held-out phrasings plus off-domain probes —
none of them appear in training.

Fact schema:

```json
{ "facts": [ { "id": "slug", "q": ["phrasing 1", "..."], "a": "canonical answer",
              "alt": ["answer variant", "..."] } ] }
```

## Pod run

`pod-train.sh` is the RunPod runbook (same setup recipe as
`tools/GpuTrainBench/pod-measure.sh`): GPU check → .NET install → model download
→ cuBLAS gate test → gen → train with checkpoints (`EPOCHS=5` default, override
with `EPOCHS=4 bash tools/FineTune/pod-train.sh`) → `probe` on the export before
anything leaves the pod. Scp the newest `out-pizza/checkpoints/*` dir first (71 MB
of adapters — enough to rebuild the model locally with `merge-probe --export`),
then `out-pizza/finetuned.gguf` if you want the pod's own file, and eval on the
laptop. Terminate the pod, don't stop it. Verified on a 4090 (v1) and a 5090 (v2,
sm_120 — ILGPU 1.5.3 compiles for it; 3.73 s/step, slower than the 4090's 3.30
because the step is custom-kernel-bound, not bandwidth-bound).

## GPU memory envelope

`GpuBackpropEngine.ArenaFloats` sizes one arena holding **every layer's forward and
backward temporaries for the whole step** (deliberate — no per-block reset), so the
activation budget scales with `layers x batch x seq` plus `batch x heads x seq²`
attention probs per layer, and it dominates the F32 weights. Measured on a 24 GB
4090 with Qwen2.5-1.5B (28L, vocab 152k): batch 8 x seq 512 OOMs, batch 8 x seq 256
OOMs (~25 GB arena by the formula), batch 2 x seq 256 fits. Scale batch/seq to the
card before renting anything.

## Known ceilings

<!-- ponytail: full-doc LM loss, no prompt masking — TrainLoop has no label masking
     today; acceptable at this scale, revisit if answers start echoing questions. -->
- Loss is computed over the whole ChatML doc (prompt tokens included), not just
  the assistant response.
- Export is F32 — the served GGUF is ~4 GB per B params. `SmmWriteOptions` has an
  F16 level that has not been tried on this path.
- `eval` is greedy single-turn; multi-turn behavior is trained (two-turn docs)
  but not yet evaluated.
