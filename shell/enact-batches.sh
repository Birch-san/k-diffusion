#!/usr/bin/env bash
set -eo pipefail -o xtrace

# rm -rf --interactive=never ~/home/shell/out
# mkdir ~/home/shell/out

JOB_ID="$(~/home/shell/batch.sh \
-t '00:30:00' \
-n 2 \
inference-multinode.sh \
--prototyping \
--log-root='/p/scratch/ccstdl/birch1/batch-log/kat_557M_2M' \
--config='configs/config_557M.jsonc' \
--ckpt='/p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors' \
--cfg-scale='1.00' \
--sampler='dpm3' \
--steps=25 \
--batch-per-gpu=5 \
--inference-n=25 \
--wds-out-root='/p/scratch/ccstdl/birch1/model-out/kat_557M_2M' \
--kdiff-dir='/p/project/ccstdl/birch1/git/k-diffusion' \
--ddp-config='/p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml')"

# exec tail -F ~/home/shell/out/{out,err}.txt

# 4 GPUS
# 4 * 5 = 20
# 2 nodes
# 2 * 4 * 5 = 40