#!/usr/bin/env bash
set -eo pipefail
# example usage:
# ~/home/shell/batch.sh -t 00:00:30

# rm -rf --interactive=never ~/home/shell/out; mkdir ~/home/shell/out && ~/home/shell/batch.sh -t '00:00:30' && tail -F ~/home/shell/out/{out,err}.txt

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

TLD="${HOSTNAME##*.}"
case "$TLD" in
  'juwels')
    venv='kdiff'
    # booster
    DEFAULT_PARTITION='develbooster' ;;
  'jureca')
    venv='kdiff-jrc'
    # dc-gpu
    DEFAULT_PARTITION='dc-gpu-devel' ;;
esac

DEFAULT_OUT_DIR="/p/project/ccstdl/birch1/shell/out"

# https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E
out="$DEFAULT_OUT_DIR/out.txt"
err="$DEFAULT_OUT_DIR/err.txt"
nodes="2"
job_name="inference"
account="cstdl"
partition="$DEFAULT_PARTITION"
time_="02:00:00"

# https://stackoverflow.com/a/28466267/5257399
while getopts o:e:n:J:A:p:t:d:-: OPT; do
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    o | out )  out="${OPTARG}" ;;
    e | err )  err="${OPTARG}" ;;
    n | nodes )  nodes="${OPTARG}" ;;
    J | job-name )  job_name="${OPTARG}" ;;
    A | account )  account="${OPTARG}" ;;
    p | partition )  partition="${OPTARG}" ;;
    t | time )  time_="${OPTARG}" ;;
    d | dependency )  dependency="${OPTARG}" ;;
    ??* )          die "Illegal option --$OPT" ;;            # bad long option
    ? )            exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

if [[ -z "$partition" ]]; then
  die "'partition' option was empty. we do try to pick a default partition based on the TLD of the caller's hostname, but only for .juwels and .jureca TLDs."
fi

if [[ "$1" == "" ]]; then
  echo >&2 "No script specified. Example: /p/project/ccstdl/birch1/shell/inference.sh" >&2
  exit 1
fi
SCRIPT="$(realpath "$1")"
SCRIPT_ARGS="${@:2}"

echo >&2 "running $SCRIPT"
echo >&2 "with args: $SCRIPT_ARGS"

# apparently srun doesn't inherit --cpus-per-task from sbatch? so we must pass it to srun explicitly
# https://apps.fz-juelich.de/jsc/hps/juwels/affinity.html#processor-affinity
CPUS_PER_TASK=64

SBATCH_VARARGS=()
if [[ -n "$dependency" ]]; then
  SBATCH_VARARGS=(${SBATCH_VARARGS[@]} "--dependency=$dependency")
fi

exec sbatch "${SBATCH_VARARGS[@]}" --parsable <<EOT
#!/bin/sh

#SBATCH -o "$out"
#SBATCH -e "$err"
#SBATCH -J "$job_name"
#SBATCH --nodes "$nodes" #Number of processors
#SBATCH -A "$account"
#SBATCH --partition "$partition"
#SBATCH --gres gpu
#SBATCH --time "$time_"
#SBATCH --exclusive
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task="$CPUS_PER_TASK"       # number of CPUs per process
#SBATCH --mem=0
##SBATCH --mem-per-gpu=32G

# JUWELS-specific (nodes have 4 GPUs)
export GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3 # ensures GPU_IDs are available with correct indicies

export MAIN_PROCESS_PORT=42069

export PYTHONFAULTHANDLER=1
export WANDB_START_METHOD="thread"

ml Stages/2024 2> /dev/null
ml Python/3.11.3
ml CUDA/12
# we use jq to write a JSON config in compute-metrics.sh.
# if this is not available on your cluster, then you could modify compute-metrics.sh
# to compose the JSON config via string templating instead.
ml jq/1.6

. "/p/project/ccstdl/birch1/venvs/$venv/bin/activate"

exec srun --cpus-per-task "$CPUS_PER_TASK" "$SCRIPT" $SCRIPT_ARGS
EOT