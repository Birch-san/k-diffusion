#!/usr/bin/env bash
# run this from the root of the k-diffusion directory, with virtual env activated

python train.py \
--config \
configs/ffhq/config_FFHQ_256px_rope.jsonc \
--out-root out \
--output-to-subdir \
--name ffhq-256-rope-b40 \
--evaluate-n 0 \
--batch-size 40 \
--demo-every 500 \
--save-every 500 \
--sample-n 36 \
--mixed-precision bf16 \
--demo-classcond-include-uncond \
--demo-img-compress \
--end-step 10000 \
--font ./kdiff_trainer/font/DejaVuSansMono.ttf \
--start-method fork \
--wandb-project hdit-ffhq-256 \
--wandb-entity mahouko \
--wandb-run-name ffhq-256-rope-b40

python train.py \
--config \
configs/ffhq/config_FFHQ_256px_additive.jsonc \
--out-root out \
--output-to-subdir \
--name ffhq-256-additive-b40 \
--evaluate-n 0 \
--batch-size 40 \
--demo-every 500 \
--save-every 500 \
--sample-n 36 \
--mixed-precision bf16 \
--demo-classcond-include-uncond \
--demo-img-compress \
--end-step 10000 \
--font ./kdiff_trainer/font/DejaVuSansMono.ttf \
--start-method fork \
--wandb-project hdit-ffhq-256 \
--wandb-entity mahouko \
--wandb-run-name ffhq-256-additive-b40