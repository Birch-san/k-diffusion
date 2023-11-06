#!/usr/bin/env bash
set -eo pipefail -o xtrace
shopt -s nullglob

JOB_START_TIME="$(date +"%Y-%m-%dT%H:%M:%S%:z")"

TLD="${HOSTNAME##*.}"
case "$TLD" in
  'juwels')
    # booster
    EVAL_PARTITION='develbooster' ;;
  'jureca')
    # dc-gpu
    EVAL_PARTITION='dc-gpu-devel' ;;
  *)
    raise "was not able to infer from your hostname's TLD which partition to schedule follow-up FID compute job"
esac

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
KDIFF_DIR="$(realpath "$SCRIPT_DIR/..")"

TRAIN_STEP='2M' # use 'unknown' if you're not sure or don't care. this is just for file naming.
MODEL_CONFIG='configs/config_557M.jsonc' # relative to k-diffusion directory
MODEL_CKPT='/p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors'
CFG_SCALE='1.00'

REALS_DATASET_CONFIG="configs/dataset/imagenet.juelich.jsonc"
DDP_CONFIG='/p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml'

SAMPLE_COUNT=50000

SCRATCH_ROOT='/p/scratch/ccstdl/birch1'
LOG_ROOT="$SCRATCH_ROOT/batch-log"
SAMPLES_OUT_ROOT="$SCRATCH_ROOT/model-out"
TMP_ROOT="$SCRATCH_ROOT/tmp"

# derive name from config. 'configs/config_557M.jsonc' -> '557M'
MODEL_NAME="$(echo "$MODEL_CONFIG" | sed -E 's#.*(^|/)([^/]*)\.jsonc?$#\2#; s/^config_//')"
echo "MODEL_NAME (inferred from MODEL_CONFIG): '$MODEL_NAME'"

JOB_QUALIFIER="$MODEL_NAME/step$TRAIN_STEP/cfg$CFG_SCALE"

WDS_OUT_DIR="$SAMPLES_OUT_ROOT/$JOB_QUALIFIER"
LOG_DIR="$LOG_ROOT/$JOB_QUALIFIER"
TMP_DIR="$TMP_ROOT/$JOB_QUALIFIER"

mkdir -p "$WDS_OUT_DIR" "$LOG_DIR" "$TMP_DIR"

echo "Copying model config '$MODEL_CONFIG' into output dir"
$(cd "$KDIFF_DIR" && cp "$MODEL_CONFIG" "$WDS_OUT_DIR/")

SAMPLER_PRESET='dpm3'
SAMPLER_STEPS=50

mkdir -p "$LOG_DIR"
# relies on nullglob
EXISTING_LOGS="$(echo "$LOG_DIR"/*.txt)"
[[ "$EXISTING_LOGS" ]] && echo "$EXISTING_LOGS" | xargs -n 1 rm

# this jq will fail if there are comments in your config! you may want to manually hardcode the image size here.
# or update it to parse using python+json5 instead.
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
--sampler="$SAMPLER_PRESET" \
--steps="$SAMPLER_STEPS" \
--batch-per-gpu=128 \
--inference-n="$SAMPLE_COUNT" \
--wds-out-dir="$WDS_OUT_DIR" \
--kdiff-dir="$KDIFF_DIR" \
--ddp-config="$DDP_CONFIG")"

RECEIPT_PREAMBLE="JOB_START_TIME: $JOB_START_TIME

MODEL_CKPT: $MODEL_CKPT
MODEL_CONFIG: $MODEL_CONFIG
TRAIN_STEP: $TRAIN_STEP
CFG_SCALE: $CFG_SCALE

WDS_OUT_DIR: $WDS_OUT_DIR
LOG_DIR: $LOG_DIR

SAMPLER_PRESET: $SAMPLER_PRESET
SAMPLER_STEPS: $SAMPLER_STEPS
SAMPLE_COUNT: $SAMPLE_COUNT

"
RECEIPT_PREAMBLE_FILE="$TMP_DIR/receipt_preamble.txt"
echo "$RECEIPT_PREAMBLE" > "$RECEIPT_PREAMBLE_FILE"

# DATASET_CONFIG_OUT_PATH="configs/dataset/pred/$JOB_QUALIFIER.jsonc"
DATASET_CONFIG_OUT_PATH="$WDS_OUT_DIR/config_dataset.jsonc"

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
--dataset-config-out-path="$DATASET_CONFIG_OUT_PATH" \
--evaluate-n="$SAMPLE_COUNT" \
--image-size="$IMAGE_SIZE" \
--kdiff-dir="$KDIFF_DIR" \
--ddp-config="$DDP_CONFIG" \
--result-description-file="$RECEIPT_PREAMBLE_FILE" \
-- \
--torchmetrics-fid \
--evaluate-with inception dinov2 \
--result-out-file "$WDS_OUT_DIR/metrics.txt")"