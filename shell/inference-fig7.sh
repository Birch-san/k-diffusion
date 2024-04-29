#!/usr/bin/env bash
set -eo pipefail -o xtrace
shopt -s nullglob

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
KDIFF_DIR="$(realpath "$SCRIPT_DIR/..")"

cd "$KDIFF_DIR"

CKPT_ROOT='/sdb/ckpt/fig7'

declare -a model_names=(
  '00_557M'
  '01_547M_8x8'
  '02_508M_16x16'
  '10_302M_mid'
  '11_295M_mid_8x8'
  '12_267M_mid_16x16'
  '20_139M_small'
  '21_134M_small_8x8'
  '22_117M_small_16x16'
)

for ix in "${!model_names[@]}"; do
    MODEL_NAME="${model_names[$ix]}"
    CKPT_LOC="$(find "$CKPT_ROOT/$MODEL_NAME" -type f -name '*.safetensors' | head -n 1)"
    echo "Inferencing $MODEL_NAME from $CKPT_LOC"
    python train.py \
--config "configs/fig7/attempt3/$MODEL_NAME.jsonc" \
--resume-inference "$CKPT_LOC" \
--inference-only \
--sample-n 256 \
--inference-out-target imgdir \
--inference-out-root "$KDIFF_DIR/out/fig7-batch/${MODEL_NAME}_candidates" \
--inference-out-seed-from 0 \
--inference-schedule "277:103;289:103;270:63;266:9;218:0;187:23;625:26;825:23;852:180;985:19;980:20;951:134;930:153;883:7;387:212;259:868;270:234;281:196;270:175;30:89;555:8;417:237;607:704,549,535" \
--demo-steps 50 \
--sampler-preset dpm2 \
--start-method fork \
--name "fig7-$MODEL_NAME" \
--mixed-precision bf16
done