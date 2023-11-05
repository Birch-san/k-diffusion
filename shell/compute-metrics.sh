#!/usr/bin/env bash
set -eo pipefail

# example invocation:
# ./compute-metrics.sh --wds-in-dir=/p/scratch/ccstdl/birch1/model-out/557M/step2M/cfg1.00 --config-target=configs/dataset/imagenet.juelich.jsonc --log-dir=/p/scratch/ccstdl/birch1/batch-log/557M/step2M/cfg1.00 --dataset-config-out-path=configs/dataset/pred/557M/step2M/cfg1.00.jsonc --image-size=256 --kdiff-dir=/p/project/ccstdl/birch1/git/k-diffusion --ddp-config=/p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml -- --torchmetrics-fid --evaluate-with inception dinov2

echo "slurm proc $SLURM_PROCID started $0"
echo "received args: $@"
# exec python -c 'import os; print(os.cpu_count()); print(len(os.sched_getaffinity(0)))'

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# https://stackoverflow.com/a/28466267/5257399
while getopts i:t:o:d:n:b:s:k:d:r:-: OPT; do
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    i | wds-in-dir )    wds_in_dir="${OPTARG}" ;;
    t | config-target ) config_target="${OPTARG}" ;; # relative to k-diffusion dir
    o | log-dir )       log_dir="${OPTARG}" ;;
    d | dataset-config-out-path ) dataset_config_out_path="${OPTARG}" ;; # relative to k-diffusion dir
    n | evaluate-n )    evaluate_n="${OPTARG}" ;;
    b | batch-per-gpu ) batch_per_gpu="${OPTARG}" ;;
    s | image-size )    image_size="${OPTARG}" ;;
    k | kdiff-dir )     kdiff_dir="${OPTARG}" ;;
    d | ddp-config )    ddp_config="${OPTARG}" ;;
    # we are forced to use a file here instead of a string, because this getopts parsing doesn't seem to handle multi-line string args
    r | result-description-file )    result_description_file="${OPTARG}" ;;
    ??* )          die "Illegal option --$OPT" ;;            # bad long option
    ? )            exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

if [[ -n "$result_description_file" ]]; then
  if [[ -f "$result_description_file" ]]; then
    RESULT_DESCRIPTION="$(cat "$result_description_file")"
  else
    die "'result-description-file' option ("$result_description_file") was not a valid file."
  fi
else
  RESULT_DESCRIPTION=''
fi
echo "result description: $RESULT_DESCRIPTION"

echo "varargs: $@"

echo "options parsed successfully. checking required args."

if [[ -z "$wds_in_dir" ]]; then
  die "'wds-in-dir' option was empty. example: /p/scratch/ccstdl/birch1/model-out/557M/step2M/cfg1.00"
fi

if [[ -z "$config_target" ]]; then
  die "'config-target' option was empty. example: configs/dataset/imagenet.juelich.jsonc"
fi

if [[ -z "$log_dir" ]]; then
  die "'log-dir' option was empty. example: /p/scratch/ccstdl/birch1/batch-log/557M/step2M/cfg1.00"
fi

if [[ -z "$dataset_config_out_path" ]]; then
  die "'dataset-config-out-path' option was empty. example: configs/dataset/pred/557M/step2M/cfg1.00.jsonc"
fi

if [[ -z "$image_size" ]]; then
  die "'image-size' option was empty. example: 256"
fi

if [[ -z "$kdiff_dir" ]]; then
  die "'kdiff-dir' option was empty. example: /p/project/ccstdl/birch1/git/k-diffusion"
fi

if [[ -z "$ddp_config" ]]; then
  die "'ddp-config' option was empty. example: /p/home/jusers/birch1/juwels/.cache/huggingface/accelerate/ddp.yaml"
fi

echo "all required args found."

: "${evaluate_n:=50000}"
: "${batch_per_gpu:=1000}"
: "${evaluate_with:=dinov2}"

set +e
TAR_LINES="$(find -L "$wds_in_dir" -maxdepth 1 -type f -name '*.tar' -execdir basename {} .tar ';' | grep -E '^[0-9]+$' | sort)"
set -e

if [[ "$TAR_LINES" == '' ]]; then
  die "Failed to find tar files inside $wds_in_dir"
fi
FIRST_TAR="$(echo "$TAR_LINES" | head -n 1)"
LAST_TAR="$(echo "$TAR_LINES" | tail -n 1)"

if [[ "$LAST_TAR" == "$FIRST_TAR" ]]; then
  TARS="$FIRST_TAR.tar"
else
  TARS="{$FIRST_TAR..$LAST_TAR}.tar"
fi

CONFIG_TEMPLATE='{
  "model": {
    "type": "none",
    "input_size": [$imageSize, $imageSize]
  },
  "dataset": {
    "type": "wds-class",
    "class_cond_key": "cls.txt",
    "wds_image_key": "img.png",
    "location": $wdsPattern,
    "num_classes": 1000,
    "classes_to_captions": "imagenet-1k"
  }
}'

set -o xtrace

CONFIG_JSON="$(jq -n \
--argjson 'imageSize' "$image_size" \
--arg 'wdsPattern' "$wds_in_dir/$TARS" \
"$CONFIG_TEMPLATE")"

cd "$kdiff_dir"

mkdir -p "$(dirname "$dataset_config_out_path")" "$log_dir"

echo "$CONFIG_JSON" > "$dataset_config_out_path"

: "${SLURM_PROCID:=0}"
: "${GPUS_PER_NODE:=4}"
: "${SLURM_JOB_NUM_NODES:=1}"
NUM_PROCESSES="$(( "$GPUS_PER_NODE" * "$SLURM_JOB_NUM_NODES" ))"

OUT_TXT="$log_dir/compute-metrics.out.$SLURM_PROCID.txt"
ERR_TXT="$log_dir/compute-metrics.err.$SLURM_PROCID.txt"

echo "writing output to: $OUT_TXT"
echo "writing errors to: $ERR_TXT"

ACC_VARARGS=()
if [[ -n "$MAIN_PROCESS_PORT" ]]; then
  ACC_VARARGS=(
    ${ACC_VARARGS[@]}
    --machine_rank "$SLURM_PROCID"
    --main_process_ip "$SLURM_LAUNCH_NODE_IPADDR"
    --main_process_port "$MAIN_PROCESS_PORT"
  )
fi

# we avoid using accelerate because multi-GPU inference seems to exhaust the dataloader prematurely (only 20k of 50k samples found)
# K_DIFFUSION_USE_COMPILE=0 exec python -m accelerate.commands.launch \
# --num_processes "$NUM_PROCESSES" \
# --num_machines "$SLURM_JOB_NUM_NODES" \
# "${ACC_VARARGS[@]}" \
# --config_file "$ddp_config" \
K_DIFFUSION_USE_COMPILE=0 exec python \
compute_metrics.py \
--config-pred "$dataset_config_out_path" \
--config-target "$config_target" \
--batch-size "$batch_per_gpu" \
--evaluate-n "$evaluate_n" \
--result-description "$RESULT_DESCRIPTION" \
"$@" \
>"$OUT_TXT" 2>"$ERR_TXT"