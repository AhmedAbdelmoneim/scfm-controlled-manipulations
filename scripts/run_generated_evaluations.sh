#!/usr/bin/env bash
# Run generated dataset evaluation configs, continuing after per-dataset failures.
#
# Usage:
#   scripts/run_generated_evaluations.sh
#
# Optional:
#   CONFIG_GLOB='configs/generated/normalized_datasets/atlases__*.yaml configs/generated/normalized_datasets/sceval__*.yaml'
#   STOP_ON_FAILURE=1
#   FORCE_RERUN=1

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_GLOB="${CONFIG_GLOB:-configs/generated/normalized_datasets/atlases__*.yaml configs/generated/normalized_datasets/sceval__*.yaml}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

shopt -s nullglob
CONFIGS=()
# shellcheck disable=SC2086
for cfg in $CONFIG_GLOB; do
    CONFIGS+=("$cfg")
done
shopt -u nullglob

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "ERROR: No generated dataset configs found for CONFIG_GLOB=${CONFIG_GLOB}"
    exit 1
fi

FAILED=()
SKIPPED=()
START_SECONDS="$(date +%s)"

config_complete() {
    local cfg="$1"
    uv run python - "$cfg" <<'PY'
from pathlib import Path
import sys

import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

results_dir = Path(cfg["results_dir"])
models = list(cfg.get("models") or [])
if not models:
    raise SystemExit(1)

missing = []
for model in models:
    path = results_dir / "evaluation" / f"{model}_metrics.csv"
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(path))

if missing:
    raise SystemExit(1)
PY
}

echo "============================================================"
echo "Starting generated evaluation batch"
echo "Configs (${#CONFIGS[@]}):"
printf '  %s\n' "${CONFIGS[@]}"
echo "Stop on failure: ${STOP_ON_FAILURE}"
echo "Force rerun: ${FORCE_RERUN}"
echo "============================================================"

for index in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$index]}"
    echo "------------------------------------------------------------"
    echo "Dataset config $((index + 1))/${#CONFIGS[@]}: ${cfg}"
    echo "Command: make evaluate CONFIG=${cfg}"
    echo "Started: $(date --iso-8601=seconds)"
    echo "------------------------------------------------------------"

    if [ "${FORCE_RERUN}" != "1" ] && config_complete "${cfg}"; then
        echo "Skipping completed config: ${cfg}"
        SKIPPED+=("${cfg}")
        continue
    fi

    if make evaluate CONFIG="${cfg}"; then
        echo "Completed: ${cfg}"
    else
        status="$?"
        echo "ERROR: Evaluation failed for ${cfg} (exit ${status})"
        FAILED+=("${cfg}")
        if [ "${STOP_ON_FAILURE}" = "1" ]; then
            break
        fi
    fi
done

END_SECONDS="$(date +%s)"
ELAPSED_SECONDS="$((END_SECONDS - START_SECONDS))"

echo "================================================------------"
echo "Generated evaluation batch finished in ${ELAPSED_SECONDS}s"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "SKIPPED completed configs (${#SKIPPED[@]}):"
    printf '  %s\n' "${SKIPPED[@]}"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED configs (${#FAILED[@]}):"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All generated configs completed successfully"
echo "============================================================"
