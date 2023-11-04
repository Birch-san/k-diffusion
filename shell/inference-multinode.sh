#!/usr/bin/env bash
set -eo pipefail

# example invocation:
# ./inference-multinode.sh --log-dir=/p/scratch/ccstdl/birch1/batch-log/557M/step2M/cfg1.00 --config=configs/config_557M.jsonc --ckpt=/p/scratch/ccstdl/birch1/ckpt/imagenet_test_v2_007_02000000.safetensors --cfg-scale=1.00 --wds-out-dir=/p/scratch/ccstdl/birch1/model-out/557M/step2M/cfg1.00 --kdiff-dir=/p/project/ccstdl/birch1/git/k-diffusion --ddp-config=/p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml

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
    o | log-dir )       log_dir="${OPTARG}" ;;
    c | config )        config="${OPTARG}" ;;
    C | ckpt )          ckpt="${OPTARG}" ;;
    s | cfg-scale )     cfg_scale="${OPTARG}" ;;
    S | sampler )       sampler="${OPTARG}" ;;
    i | steps )         steps="${OPTARG}" ;;
    b | batch-per-gpu ) batch_per_gpu="${OPTARG}" ;;
    n | inference-n )   inference_n="${OPTARG}" ;;
    w | wds-out-dir )   wds_out_dir="${OPTARG}" ;;
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

if [[ -z "$wds_out_dir" ]]; then
  die "'wds-out-dir' option was empty. example: /p/scratch/ccstdl/birch1/model-out/557M/step2M/cfg1.00"
fi

if [[ -z "$log_dir" ]]; then
  die "'log-dir' option was empty. example: /p/scratch/ccstdl/birch1/batch-log/557M/step2M/cfg1.00"
fi

if [[ -z "$kdiff_dir" ]]; then
  die "'kdiff-dir' option was empty. example: /p/project/ccstdl/birch1/git/k-diffusion"
fi

if [[ -z "$ddp_config" ]]; then
  die "'ddp-config' option was empty. example: /p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml"
fi

echo "all required args found."

: "${cfg_scale:='1.00'}"
: "${steps:=50}"
: "${sampler:=dpm3}"
: "${batch_per_gpu:=128}"

: "${inference_n:=50000}"

if [[ "$prototyping" == "true" ]]; then
  # get results a few seconds faster by skipping compile.
  export K_DIFFUSION_USE_COMPILE=0
fi

mkdir -p "$wds_out_dir" "$log_dir"

OUT_TXT="$log_dir/inference.out.$SLURM_PROCID.txt"
ERR_TXT="$log_dir/inference.err.$SLURM_PROCID.txt"

echo "log_dir: $log_dir"
echo "writing output to: $OUT_TXT"
echo "writing errors to: $ERR_TXT"

NUM_PROCESSES="$(( "$GPUS_PER_NODE" * "$SLURM_JOB_NUM_NODES" ))"
CUMULATIVE_BATCH="$(( "$batch_per_gpu" * "$NUM_PROCESSES" ))"

echo "ckpt: $ckpt"
echo "config: $config"
echo "wds_out_dir: $wds_out_dir"
echo "cfg_scale: $cfg_scale"
echo "sampler: $sampler"
echo "steps: $steps"
echo "inference_n: $inference_n"
echo "kdiff_dir: $kdiff_dir"

echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NUM_PROCESSES: $NUM_PROCESSES"

set -o xtrace
cd "$kdiff_dir"

exec python -m accelerate.commands.launch \
--num_processes "$NUM_PROCESSES" \
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
--cfg-scale "$cfg_scale" \
--inference-only \
--demo-steps "$steps" \
--sampler-preset "$sampler" \
--sample-n "$CUMULATIVE_BATCH" \
--inference-n "$inference_n" \
--inference-out-wds-root "$wds_out_dir" \
>"$OUT_TXT" 2>"$ERR_TXT"