#!/usr/bin/env bash
set -eo pipefail -o xtrace

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
KDIFF_DIR="$(realpath "$SCRIPT_DIR/..")"

TRAIN_STEP='2M' # use 'unknown' if you're not sure or don't care. this is just for file naming.
MODEL_CONFIG='configs/config_557M.jsonc' # relative to k-diffusion directory
MODEL_CKPT='/p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors'
CFG_SCALE='1.00'

REALS_DATASET_CONFIG="configs/dataset/imagenet.juelich.jsonc"
DDP_CONFIG='/p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml'

SAMPLE_COUNT=50000

LOG_ROOT='/p/scratch/ccstdl/birch1/batch-log'
SAMPLES_OUT_ROOT='/p/scratch/ccstdl/birch1/model-out'

# derive name from config. 'configs/config_557M.jsonc' -> '557M'
MODEL_NAME="$(echo "$MODEL_CONFIG" | sed -E 's#.*(^|/)([^/]*)\.jsonc?$#\2#; s/^config_//')"
echo "MODEL_NAME (inferred from MODEL_CONFIG): '$MODEL_NAME'"

JOB_QUALIFIER="$MODEL_NAME/step$TRAIN_STEP/cfg$CFG_SCALE"

WDS_OUT_DIR="$SAMPLES_OUT_ROOT/$JOB_QUALIFIER"
LOG_DIR="$LOG_ROOT/$JOB_QUALIFIER"

IMAGE_SIZE="$(cd "$KDIFF_DIR" && jq -r '.model.input_size[0]' <"$MODEL_CONFIG")"
if [[ ! "$IMAGE_SIZE" =~ ^[0-9]+$ ]]; then
  die "Received non-integer image size from model config '$MODEL_CONFIG' (via jq query '.model.input_size[0]'): '$IMAGE_SIZE'"
fi
echo "IMAGE_SIZE (inferred from parsing MODEL_CONFIG): '$IMAGE_SIZE'"

INFERENCE_JOB_ID="$("$SCRIPT_DIR/batch.sh" \
-o "$LOG_DIR/inference-srun.out.txt" \
-e "$LOG_DIR/inference-srun.err.txt" \
-p dc-gpu \
--job-name='inference' \
-t '01:00:00' \
-n 5 \
"$SCRIPT_DIR/inference-multinode.sh" \
--log-dir="$LOG_DIR" \
--config="$MODEL_CONFIG" \
--ckpt="$MODEL_CKPT" \
--cfg-scale="$CFG_SCALE" \
--sampler='dpm3' \
--steps=50 \
--batch-per-gpu=128 \
--inference-n="$SAMPLE_COUNT" \
--wds-out-dir="$WDS_OUT_DIR" \
--kdiff-dir="$KDIFF_DIR" \
--ddp-config="$DDP_CONFIG")"

METRICS_JOB_ID="$("$SCRIPT_DIR/batch.sh" \
-o "$LOG_DIR/compute-metrics-srun.out.txt" \
-e "$LOG_DIR/compute-metrics-srun.err.txt" \
-n 1 \
--job-name='evaluate-samples' \
-t '00:30:00' \
--dependency="afterok:$INFERENCE_JOB_ID" \
"$SCRIPT_DIR/compute-metrics.sh" \
--wds-in-dir="$WDS_OUT_DIR" \
--config-target="$REALS_DATASET_CONFIG" \
--log-dir="$LOG_DIR" \
--dataset-config-out-path="configs/dataset/pred/$JOB_QUALIFIER.jsonc" \
--evaluate-n="$SAMPLE_COUNT" \
--evaluate-with='dinov2' \
--image-size="$IMAGE_SIZE" \
--kdiff-dir="$KDIFF_DIR" \
--torchmetrics-fid \
--ddp-config="$DDP_CONFIG")"