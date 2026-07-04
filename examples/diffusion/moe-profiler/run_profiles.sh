#!/usr/bin/env bash
# run_profiles.sh — profile MoE expert usage per workload with llama-diffusion-cli
#
# Prompt layout (one prompt per .txt file, directory name = workload tag):
#   prompts/
#     bengali/  01.txt 02.txt ...
#     coding/   01.txt 02.txt ...
#     english/  ...
#     math/     ...
#
# Usage:
#   ./run_profiles.sh <llama-diffusion-cli> <model.gguf> <prompts_dir> [extra cli args...]
#
# Example:
#   ./run_profiles.sh ../llama.cpp/build/bin/llama-diffusion-cli \
#       ../diffusiongemma-26B-A4B-it-Q4_K_M.gguf prompts -ngl 99 -sm none -mg 0 -ncmoe 16 -c 4096 -n 512
#
# Env overrides:
#   MOE_PROFILE_OUT       output csv (default moe_profile.csv, appended)
#   MOE_PROFILE_MIN_NTOK  e.g. 256 to count only large (canvas-sized) evals

set -euo pipefail

BIN=$1; MODEL=$2; PDIR=$3; shift 3
OUT=${MOE_PROFILE_OUT:-moe_profile.csv}

for tagdir in "$PDIR"/*/; do
    tag=$(basename "$tagdir")
    for f in "$tagdir"*.txt; do
        [ -e "$f" ] || continue
        echo ">>> [$tag] $f"
        MOE_PROFILE=1 \
        MOE_PROFILE_TAG="$tag" \
        MOE_PROFILE_OUT="$OUT" \
        MOE_PROFILE_MIN_NTOK="${MOE_PROFILE_MIN_NTOK:-0}" \
            "$BIN" -m "$MODEL" -p "$(cat "$f")" "$@" < /dev/null
    done
done

echo
echo "done -> $OUT"
echo "next:   python3 analyze_experts.py $OUT --plot"
