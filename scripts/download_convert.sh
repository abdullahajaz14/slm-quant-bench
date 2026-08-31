#!/usr/bin/env bash
# scripts/download_convert.sh <model-key> [--keep-weights]
#
# Operational plumbing (like the configs, this is tooling, not framework
# logic): downloads instruct weights from HF, converts to FP16 GGUF,
# quantises to Q8_0 and Q4_K_M, prints sizes.
#
# Prereqs, one-time:
#   1. huggingface-cli login          (gated models: accept licence on HF)
#   2. llama.cpp checkout + build:
#        git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
#        cd ~/llama.cpp && cmake -B build && cmake --build build -j
#      (Metal is enabled automatically on macOS)
#   3. pip install "huggingface_hub[cli]"
#
# Env: LLAMACPP_DIR (default ~/llama.cpp). Run from the repo root.

set -euo pipefail

MODEL_KEY="${1:?usage: download_convert.sh <model-key> [--keep-weights]}"
KEEP="${2:-}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
QUANTIZE="$LLAMACPP_DIR/build/bin/llama-quantize"
CONVERT="$LLAMACPP_DIR/convert_hf_to_gguf.py"

case "$MODEL_KEY" in
  phi3-mini)  REPO="microsoft/Phi-3-mini-4k-instruct" ;;
  gemma2-2b)  REPO="google/gemma-2-2b-it" ;;
  llama32-3b) REPO="meta-llama/Llama-3.2-3B-Instruct" ;;
  mistral7b)  REPO="mistralai/Mistral-7B-Instruct-v0.3" ;;
  *) echo "unknown model key: $MODEL_KEY"; exit 1 ;;
esac

# Disk guard: FP16 + weights for the 7B needs ~30 GB transiently.
FREE_GB=$(df -g . 2>/dev/null | awk 'NR==2{print $4}' || df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${FREE_GB:-0}" -lt 25 ]; then
  echo "WARNING: only ${FREE_GB} GB free; spec says keep 25 GB headroom." >&2
fi

mkdir -p weights models

# The CLI was renamed huggingface-cli -> hf in hub v1.x; support both so
# the script runs unchanged on the device and on Colab.
if command -v hf > /dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli > /dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "no Hugging Face CLI found: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "==> downloading $REPO (with $HF_CLI)"
# Exclude duplicate formats: convert_hf_to_gguf.py reads safetensors, and
# some repos also ship .pth/.bin/GGUF copies that would double the download.
# NOTE: `hf download REPO [FILENAMES]...` treats bare patterns as files
# to download, so every exclusion needs its own --exclude flag.
"$HF_CLI" download "$REPO" --local-dir "weights/$MODEL_KEY" \
  --exclude "original/*" --exclude "*.pth" --exclude "*.gguf"

if [ ! -f "weights/$MODEL_KEY/config.json" ]; then
  echo "ERROR: download produced no config.json in weights/$MODEL_KEY" >&2
  exit 1
fi

# Every artefact is written under a .part name and moved into place
# only once its producer has exited successfully, so an interrupted
# conversion can never leave a file at the name the rest of the
# pipeline treats as finished. This is not hypothetical: a Colab
# session dropped during Mistral's Q4_K_M quantisation at tensor 136 of
# 291, and the notebook's "skip if the file exists" guard would have
# accepted that truncated artefact, passed its own completeness check,
# and run the entire sweep against a corrupt model while reporting
# nothing unusual.
echo "==> converting to FP16 GGUF"
rm -f "models/$MODEL_KEY-fp16.gguf.part"
python "$CONVERT" "weights/$MODEL_KEY" \
  --outfile "models/$MODEL_KEY-fp16.gguf.part" --outtype f16
mv "models/$MODEL_KEY-fp16.gguf.part" "models/$MODEL_KEY-fp16.gguf"

echo "==> quantising Q8_0 and Q4_K_M"
rm -f "models/$MODEL_KEY-q8_0.gguf.part" "models/$MODEL_KEY-q4_k_m.gguf.part"
"$QUANTIZE" "models/$MODEL_KEY-fp16.gguf" \
  "models/$MODEL_KEY-q8_0.gguf.part"  Q8_0
mv "models/$MODEL_KEY-q8_0.gguf.part" "models/$MODEL_KEY-q8_0.gguf"
"$QUANTIZE" "models/$MODEL_KEY-fp16.gguf" \
  "models/$MODEL_KEY-q4_k_m.gguf.part" Q4_K_M
mv "models/$MODEL_KEY-q4_k_m.gguf.part" "models/$MODEL_KEY-q4_k_m.gguf"

if [ "$KEEP" != "--keep-weights" ]; then
  echo "==> removing raw weights (pass --keep-weights to skip)"
  rm -rf "weights/$MODEL_KEY"
fi

echo "==> done:"
ls -lh models/$MODEL_KEY-*.gguf
echo "Record these sizes and sha256 sums in the README (Chapter 3 promise):"
shasum -a 256 models/$MODEL_KEY-*.gguf
