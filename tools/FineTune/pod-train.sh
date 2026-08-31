#!/usr/bin/env bash
# CARD-1404: PizzaMind training run on a rented CUDA pod (RunPod 4090 class). Idempotent.
# Run from the REPO ROOT of a full checkout on the demo/pizza-expert branch:
#   bash tools/FineTune/pod-train.sh [Qwen/Qwen2.5-1.5B-Instruct-GGUF fp16 by default]
#
# Same GPU-check / libcuda.so-symlink / .NET-install recipe as tools/GpuTrainBench/pod-measure.sh.
# Steps: gate test (cuBLAS on real CUDA — do NOT skip) -> dataset gen -> LoRA train -> the
# finetuned.gguf lands in out-pizza/. scp that file home; the eval runs on the laptop.
set -euo pipefail

MODEL_URL="${1:-https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-fp16.gguf}"
MODEL="/workspace/$(basename "$MODEL_URL")"
# v1 (58k tokens) needed 12 epochs to memorize; v2 (~400k tokens, ~780 steps/epoch, ~40 min
# each on a 4090) plans 4-6. Override: EPOCHS=4 bash tools/FineTune/pod-train.sh
EPOCHS="${EPOCHS:-5}"
[ -f "SharpMind.sln" ] || { echo "Run this from the repo root (SharpMind.sln not found here)." >&2; exit 1; }

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# ponytail: container toolkits mount libcuda.so.1 but not always the dev symlink ILGPU dlopen()s.
if ! ldconfig -p | grep -q 'libcuda\.so '; then
  real=$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')
  [ -n "${real:-}" ] && ln -sf "$real" "$(dirname "$real")/libcuda.so" && ldconfig && echo "linked libcuda.so -> $real"
fi
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== .NET"
export DOTNET_ROOT="${DOTNET_ROOT:-/workspace/dotnet}"
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
if ! command -v dotnet >/dev/null 2>&1 || ! dotnet --list-sdks | grep -q '^10\.'; then
  curl -sSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
  bash /tmp/dotnet-install.sh --channel 10.0 --install-dir "$DOTNET_ROOT"
fi
dotnet --version

echo "== Model"
[ -f "$MODEL" ] || curl -L --fail -o "$MODEL" "$MODEL_URL"
ls -lh "$MODEL"

echo "== Step 1/3: gate test — cuBLAS GEMM against the double-precision reference"
dotnet test SharpMind.Tests/SharpMind.Tests.csproj -c Release \
  --filter "FullyQualifiedName~SharpMind.Tests.GPU.GemmTests.Gemm_AllLayouts_MatchDoubleReference"

LOG="pod-train-$(hostname)-$(date +%Y%m%d-%H%M%S).log"
{
echo "== Step 2/3: dataset"
dotnet run -c Release --project tools/FineTune -- gen tools/FineTune/pizza/facts out-pizza

echo "== Step 3/3: LoRA fine-tune (checkpoints -> out-pizza/checkpoints)"
# Shape note (measured 2026-08-24 on a 24 GB 4090): GpuBackpropEngine's arena keeps every
# layer's fwd+bwd temporaries resident per step (see ArenaFloats), so on 24 GB a 1.5B model
# OOMs at batch 8 x seq 512 AND batch 8 x seq 256; batch 2 x seq 256 fits comfortably.
dotnet run -c Release --project tools/FineTune -- train "$MODEL" out-pizza/train.jsonl \
  --gpu --rank 16 --epochs "$EPOCHS" --batch 2 --seq 256 --lr 1e-4 --out out-pizza --ckpt-interval 300
echo "== Probe: 30-second sanity check BEFORE anything leaves the pod"
dotnet run -c Release --project tools/FineTune -- probe out-pizza/finetuned.gguf \
  "Someone told me pinsa is an ancient Roman recipe. True?" 60
} 2>&1 | tee "$LOG"

echo
echo "If the probe printed coherent English, scp back: out-pizza/finetuned.gguf, the"
echo "newest out-pizza/checkpoints/* dir (recovery insurance), and $LOG"
echo "  scp root@<pod>:$(pwd)/out-pizza/finetuned.gguf ."
echo "Then TERMINATE the pod (not just Stop — a stopped pod still bills its disk)."
