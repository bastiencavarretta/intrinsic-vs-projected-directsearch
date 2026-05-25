#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Pass --dryrun to run with 2 instances and budget 2 (config unchanged, results saved with _dryrun suffix)
DRYRUN=${1:+--dryrun}

if [ -n "$DRYRUN" ]; then
    latest_pkl() { ls "$1/results/"*.pkl | grep "_dryrun" | tail -1; }
else
    latest_pkl() { ls "$1/results/"*.pkl | grep -v "_dryrun" | tail -1; }
fi

# ── Sections 5 & 6.2 — Dimension influence ──────────────────────────────
EXP_DI="experiments/finalrun_budg100mdimcodimall_v1"
EXP_DI_Budg200="experiments/finalrun_budg200mdim32codim4_uselessother_v1"
cd "$SCRIPT_DIR/dimension_influence"

python run.py --exppath $EXP_DI $DRYRUN
python run.py --exppath $EXP_DI_Budg200 $DRYRUN

python plot_dataprofile.py --exppath $EXP_DI $DRYRUN
python plot_dataprofile.py --exppath $EXP_DI_Budg200 $DRYRUN

python performancetables.py $(latest_pkl $EXP_DI) --save --savedir $EXP_DI/tables/ $DRYRUN
python performancetables.py $(latest_pkl $EXP_DI_Budg200) --save --savedir $EXP_DI_Budg200/tables/ $DRYRUN

# ── Section 6.3 — Synchronization of rotations ──────────────────────────
EXP_SR="experiments/k5_n2_budg500_noise1em6_nowarmstart_v1"
cd "$SCRIPT_DIR/synchronization_rotations"

python run.py --exppath $EXP_SR $DRYRUN
python plot_dataprofile.py --exppath $EXP_SR $DRYRUN
