#!/usr/bin/env bash
set -eo pipefail -o xtrace

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

WDS_OUT_ROOT='/p/scratch/ccstdl/birch1/model-out/kat_557M_2M'
JOB_ID="$("$SCRIPT_DIR/batch.sh" \
-t '00:15:00' \
-n 2 \
"$SCRIPT_DIR/inference-multinode.sh" \
--prototyping \
--log-root='/p/scratch/ccstdl/birch1/batch-log/kat_557M_2M' \
--config='configs/config_557M.jsonc' \
--ckpt='/p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors' \
--cfg-scale='1.00' \
--sampler='dpm3' \
--steps=25 \
--batch-per-gpu=5 \
--inference-n=70 \
--wds-out-root="$WDS_OUT_ROOT" \
--kdiff-dir='/p/project/ccstdl/birch1/git/k-diffusion' \
--ddp-config='/p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml')"

srun --nodes=1 \
-o /p/scratch/ccstdl/birch1/batch-log/test-jobdep-srun.out.txt \
-e /p/scratch/ccstdl/birch1/batch-log/test-jobdep-srun.err.txt \
-A cstdl \
--partition "$EVAL_PARTITION" \
--gres gpu \
--job-name=evaluate-samples \
--exclusive \
--threads-per-core=1 \
--cpus-per-task=64 \
--mem=0 \
--time=00:05:00 \
--dependency="afterok:$JOB_ID" \
"$SCRIPT_DIR/test-jobdep.sh" \
--wds-out-root="$WDS_OUT_ROOT"