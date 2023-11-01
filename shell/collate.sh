#!/usr/bin/env bash
# ./collate.sh /p/scratch/ccstdl/birch1/model-out/kat_557M 1.00
if [[ "$1" == "" ]]; then
  echo "No input root provided. Example: /p/scratch/ccstdl/birch1/model-out/kat_557M" >&2
  exit 1
fi
if [[ "$2" == "" ]]; then
  echo "No CFG scale. Example: 1.00" >&2
  exit 1
fi
ROOT="$1"
CFG="$2"
COLLATE_OUT="$ROOT/collated/cfg$2"
mkdir -p "$COLLATE_OUT"
echo "created $COLLATE_OUT"
exec find "$ROOT/cfg$CFG" -maxdepth 2 -mindepth 2 -type f -name '*.tar' | sort | awk "{ printf(\"-s %s $COLLATE_OUT/%05d.tar\n\", \$0, NR-1) }" | xargs -n 3 ln