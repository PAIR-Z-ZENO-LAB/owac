#!/usr/bin/env bash
set -euo pipefail
method=${1:?dino-corp, rawvla, or rawild}
config=${2:-configs/experiments/real_fruit_baselines_20261120.json}
source_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$source_root"
isp_run=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["isp_run"])' "$config")
source "$isp_run/environment.sh"
export PYTHONPATH="$source_root/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
exec "$isp_run/runtime/bin/python" -u -m owac.training.fruit_baseline_train --method "$method" --config "$config" "${@:3}"
