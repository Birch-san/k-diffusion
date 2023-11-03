#!/usr/bin/env bash
set -eo pipefail

# example invocation:
# ./inference-multinode.sh --ckpt=checkpoint --config=config --wds_out_root=wds_out_root --cfg_scale=1.00
# ./inference-multinode.sh --ckpt=/p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors --config=configs/config_557M.jsonc --wds_out_root=/p/scratch/ccstdl/birch1/model-out/kat_557M_2M --cfg_scale=1.00

echo "slurm proc $SLURM_PROCID started $0"
echo "received args: $@"
# exec python -c 'import os; print(os.cpu_count()); print(len(os.sched_getaffinity(0)))'

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# https://stackoverflow.com/a/28466267/5257399
while getopts po:c:C:s:S:b:i:n:w:k:d:-: OPT; do
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    p | prototyping )   prototyping=true ;;
    o | log-root )      log_root="${OPTARG}" ;;
    c | config )        config="${OPTARG}" ;;
    C | ckpt )          ckpt="${OPTARG}" ;;
    s | cfg-scale )     cfg_scale="${OPTARG}" ;;
    S | sampler )       sampler="${OPTARG}" ;;
    i | steps )         steps="${OPTARG}" ;;
    b | batch-per-gpu ) batch_per_gpu="${OPTARG}" ;;
    n | inference-n )   inference_n="${OPTARG}" ;;
    w | wds-out-root )  wds_out_root="${OPTARG}" ;;
    k | kdiff-dir )     kdiff_dir="${OPTARG}" ;;
    d | ddp-config )    ddp_config="${OPTARG}" ;;
    ??* )          die "Illegal option --$OPT" ;;            # bad long option
    ? )            exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

echo "options parsed successfully. checking required args."

if [[ -z "$config" ]]; then
  die "'config' option was empty. example: configs/config_557M.jsonc"
fi

if [[ -z "$ckpt" ]]; then
  die "'ckpt' option was empty. example: /p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors"
fi

if [[ -z "$wds_out_root" ]]; then
  die "'wds-out-root' option was empty. example: /p/scratch/ccstdl/birch1/model-out/kat_557M_2M"
fi

if [[ -z "$log_root" ]]; then
  die "'log-root' option was empty. example: /p/scratch/ccstdl/birch1/batch-log/kat_557M_2M"
fi

if [[ -z "$kdiff_dir" ]]; then
  die "'kdiff-dir' option was empty. example: /p/project/ccstdl/birch1/git/k-diffusion"
fi

if [[ -z "$ddp_config" ]]; then
  die "'ddp-config' option was empty. example: /p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml"
fi

echo "all required args found."

CFG_SCALE="${cfg_scale:-'1.00'}"
STEPS="${steps:-50}"
SAMPLER="${sampler:-dpm3}"
BATCH_PER_GPU="${batch_per_gpu:-128}"

SAMPLES_TOTAL="${inference_n:-50000}"

if [[ "$prototyping" == "true" ]]; then
  # get results a few seconds faster by skipping compile.
  export K_DIFFUSION_USE_COMPILE=0
fi

WDS_OUT_DIR="$wds_out_root/cfg$CFG_SCALE"

LOG_DIR="$log_root/cfg$CFG_SCALE"
mkdir -p "$WDS_OUT_DIR" "$LOG_DIR"

OUT_TXT="$LOG_DIR/out.$SLURM_PROCID.txt"
ERR_TXT="$LOG_DIR/err.$SLURM_PROCID.txt"

echo "LOG_DIR (derived from log_root and CFG_SCALE): $LOG_DIR"
echo "writing output to: $OUT_TXT"
echo "writing errors to: $ERR_TXT"

NUM_PROCESSES="$(( "$GPUS_PER_NODE" * "$SLURM_JOB_NUM_NODES" ))"
CUMULATIVE_BATCH="$(( "$BATCH_PER_GPU" * "$NUM_PROCESSES" ))"

echo "ckpt: $ckpt"
echo "config: $config"
echo "wds_out_root: $wds_out_root"
echo "CFG_SCALE: $CFG_SCALE"
echo "SAMPLER: $SAMPLER"
echo "STEPS: $STEPS"
echo "SAMPLES_TOTAL: $SAMPLES_TOTAL"
echo "kdiff_dir: $kdiff_dir"

echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NUM_PROCESSES: $NUM_PROCESSES"

echo "WDS_OUT_DIR (derived from wds_out_root and CFG_SCALE): $WDS_OUT_DIR"

set -o xtrace
cd "$kdiff_dir"

exec python -m accelerate.commands.launch \
--num_processes="$NUM_PROCESSES" \
--num_machines "$SLURM_JOB_NUM_NODES" \
--machine_rank "$SLURM_PROCID" \
--main_process_ip "$SLURM_LAUNCH_NODE_IPADDR" \
--main_process_port "$MAIN_PROCESS_PORT" \
--config_file "$ddp_config" \
train.py \
--out-root out \
--output-to-subdir \
--config "$config" \
--name inference \
--resume-inference "$ckpt" \
--cfg-scale "$CFG_SCALE" \
--inference-only \
--sample-n "$CUMULATIVE_BATCH" \
--inference-n "$SAMPLES_TOTAL" \
--inference-out-wds-root "$WDS_OUT_DIR" \
>"$OUT_TXT" 2>"$ERR_TXT"