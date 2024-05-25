#!/usr/bin/env bash
set -eo pipefail -o xtrace
shopt -s nullglob

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
KDIFF_DIR="$(realpath "$SCRIPT_DIR/..")"

cd "$KDIFF_DIR"

for i in out/ffhq-256-{rope,additive}-b40/ckpt/*.safetensors; do # Whitespace-safe but not recursive.
    echo "$i"
    ABLATION_TYPE="$(echo "$i" | sed 's#^out/ffhq-256-##; s#-b40.*$##')"
    STEP="$(echo "$i" | sed 's#^.*_000##; s#.safetensors$##')"

    echo "$ABLATION_TYPE $STEP"

    python train.py \
--config "configs/ffhq/config_FFHQ_256px_$ABLATION_TYPE.jsonc" \
--resume-inference "out/ffhq-256-$ABLATION_TYPE-b40/ckpt/ffhq-256-$ABLATION_TYPE-b40_000$STEP.safetensors" \
--inference-only \
--sample-n 8 \
--inference-out-target imgdir \
--inference-out-root "out/ffhq/$ABLATION_TYPE/$STEP" \
--inference-schedule :0-7 \
--demo-steps 50 \
--sampler-preset dpm3 \
--start-method fork \
--name ffhq-ablation \
--mixed-precision bf16
done