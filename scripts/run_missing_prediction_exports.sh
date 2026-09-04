#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_step() {
    local name="$1"
    local script="$2"
    echo "START ${name} $(date --iso-8601=seconds)"
    .venv/bin/python -u "scripts/${script}" > "results/logs_${name}.txt" 2>&1
    echo "DONE ${name} $(date --iso-8601=seconds)"
}

run_step "pvod_xgboost_export" "run_pvod_xgboost.py"
run_step "pvod_mlp_export" "run_pvod_day_ahead_matched_baselines.py"
run_step "pvod_patchtst_export" "run_pvod_patchtst.py"
run_step "emsx_gru_export" "run_emsx_gru.py"
run_step "emsx_xgboost_export" "run_emsx_xgboost.py"

echo "ALL_DONE $(date --iso-8601=seconds)"
