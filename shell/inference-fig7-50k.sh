#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
KDIFF_DIR="$SCRIPT_DIR/.."
cd "$KDIFF_DIR"

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error

if [[ -z "$HUGGINGFACE_TOKEN" ]]; then
  die "you should provide HUGGINGFACE_TOKEN env var so we can upload the inferenced samples to HF"
fi

#####
# single-node, multi-GPU script for inferencing many samples + computing FID
#####

# TODO: activate k-diffusion env

SCRATCH_DIR='/path/to/scatch'
CKPT_DIR="$SCRATCH_DIR/ckpt"

set -o xtrace

# in pixels if it's rgb, or in latents if it's latent
IMAGE_SIZE=256

model_names=(
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
# 0 to 8, inclusive
MODEL_IX=0
# used for naming the output directory and also for indexing into model configs
MODEL_NAME="${model_names[$MODEL_IX]}"



OUT_ROOT="$SCRATCH_DIR/inference-out"
OUT_DIR="$OUT_ROOT/$MODEL_NAME"

mkdir -p "$OUT_DIR"
TEST_TRANSFER="$OUT_DIR/test.txt"
# this also checks whether we have write permissions to the scratch directory
echo "Checking whether upload succeeds" > "$TEST_TRANSFER"
# test your HF token early so we don't have tragedy a few hours later
HF_NAME="fig7-${MODEL_NAME}"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type=dataset --token="$HUGGINGFACE_TOKEN" --private "$HF_NAME" "$TEST_TRANSFER" "README.md"



# relative to k-diffusion directory
# the config of the model we'd like to inference from
IN_MODEL_CONFIG="configs/fig7/attempt3/${MODEL_NAME}.jsonc"

# environment-specific config saying where imagenet-1k is on your computer
IN_REALS_CONFIG='configs/dataset/imagenet.juelich.jsonc'

ckpts=(
  "00_557m_p_4_bs_32x8/hdit_557m_p_4_test_bs_32x8_01000000.safetensors"
  "01_547M_8x8_bs_32x8/01_547M_8x8_bs_32x8_01000000.safetensors"
  "02_508M_16x16_bs_64x4/02_508M_16x16_bs_64x4_01000000.safetensors"
  "10_302M_mid_bs_32x8/10_302M_mid_bs_32x8_01000000.safetensors"
  "11_295M_mid_8x8_bs_32x8/11_295M_mid_8x8_bs_32x8_01000000.safetensors"
  "12_267M_mid_16x16_bs_64x4/12_267M_mid_16x16_bs_64x4_01000000.safetensors"
  "20_139M_small_bs_64x4/20_139M_small_bs_64x4_01000000.safetensors"
  "21_134M_small_8x8/21_134M_small_8x8_01000000.safetensors"
  "22_117M_small_16x16/22_117M_small_16x16_01000000.safetensors"
)
CKPT="/p/fastdata/mmlaion/hdit_scaling_checkpoints_alex/${ckpts[$MODEL_IX]}"

# btw increasing CFG scale above 1 effectively doubles your batch size
CFG_SCALE=1.00

BATCH_PER_GPU=256

# usual sampler preset, which we should use in most cases
SAMPLER_PRESET='dpm3'
SAMPLER_STEPS=50

MIXED_PRECISION=bf16
QUALIFIER='.mixedbf16'

# change this based on how girthy your cluster is.
GPUS_PER_NODE=4
NUM_NODES=1
NUM_PROCESSES="$(( "$GPUS_PER_NODE" * "$NUM_NODES" ))"
CUMULATIVE_BATCH="$(( "$BATCH_PER_GPU" * "$NUM_PROCESSES" ))"

N_SAMPLES=50000

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m accelerate.commands.launch \
--multi_gpu \
--num_processes "$NUM_PROCESSES" \
--num_machines "$NUM_NODES" \
train.py \
--config "$IN_MODEL_CONFIG" \
--resume-inference "$CKPT" \
--inference-only \
--sample-n "$CUMULATIVE_BATCH" \
--cfg-scale "$CFG_SCALE" \
--inference-out-wds-root "$OUT_DIR" \
--inference-n "$N_SAMPLES" \
--sampler-preset "$SAMPLER_PRESET" \
--demo-steps "$SAMPLER_STEPS" \
--mixed-precision "$MIXED_PRECISION"
echo "Finished outputting wds to $OUT_DIR"

README_PATH="$OUT_DIR/README.md"
echo "50k inferenced samples from $MODEL_NAME

Model:  
https://huggingface.co/CitationMax/${MODEL_NAME}

Config:  
https://github.com/Birch-san/k-diffusion/blob/its-not-dit/${IN_MODEL_CONFIG}
" > "$README_PATH"
echo "Output README at $README_PATH"

# upload the 50k samples to a new private dataset on HF:
HF_URL="https://huggingface.co/datasets/CitationMax/$HF_NAME"
echo "Uploading README to $HF_URL"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type=dataset --token="$HUGGINGFACE_TOKEN" --private "$HF_NAME" "$README_PATH" "README.md"
echo "Uploading .tars to $HF_URL"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type=dataset --token="$HUGGINGFACE_TOKEN" --private "$HF_NAME" "$OUT_DIR" . --include='*.tar'


#####
# let's also collate the .tars into a .npz for use in OpenAI eval.
#####

OUT_CONFIG_PATH="$OUT_DIR/dataset-out-config.json"
echo "{
  \"model\": {
    \"type\": \"none\",
    \"input_size\": [$IMAGE_SIZE, $IMAGE_SIZE]
  },
  \"dataset\": {
    \"type\": \"wds-class\",
    \"class_cond_key\": \"cls.txt\",
    \"wds_image_key\": \"img.png\",
    \"location\": \"$OUT_DIR/{00000..00004}.tar\",
    \"num_classes\": 1000,
    \"classes_to_captions\": \"imagenet-1k\",
    \"estimated_samples\": 50000
  }
}" >"$OUT_CONFIG_PATH"
echo "Output dataset config $OUT_CONFIG_PATH"

echo "Collating .tars into .npz"
OUT_NPZ="$OUT_DIR/${MODEL_NAME}-50k.npz"

# Note: this requires a lot of RAM. you can avoid OOM by using:
#   --mem-map-out "$OUT_NPZ.dat" \
# but this will double the time taken and use double the disk space (you can delete afterwards of course)
python dataset_to_npz.py \
--config "$OUT_CONFIG_PATH" \
--out "$OUT_NPZ"
echo "Output .npz at $OUT_NPZ"

echo "Uploading .npz to $HF_URL"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli upload --repo-type=dataset --token="$HUGGINGFACE_TOKEN" --private "$HF_NAME" "$OUT_DIR" . --include='*.npz'