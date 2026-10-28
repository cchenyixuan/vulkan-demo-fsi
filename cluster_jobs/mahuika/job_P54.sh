#!/bin/bash -l
#SBATCH -A uoa04509
#SBATCH -p genoa
#SBATCH --gres=gpu:pro_6000:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /nesi/nobackup/uoa04509/sph/logs/p54_%x_%j.out
# Case-1 pre-test at 54 m3 (2026-10-26, user: "都提交到mahuika吧"): haringa54, 30 mm (2,315,295 particles,
# dt 1.0119e-4 s), state_limited with E. coli-like demonstration parameters, tau_up = tau_down = 20 s, x_bio 10 g/kg,
# feed 0.44 mol/s from t = 0, 240 s = 2,371,800 steps (120 s development, 120 s statistics). COUPLING = own | mean
# (mean: population-mean mu written every 0.5 s). Lifelines 20,000 every 0.1 s (988 steps), probes 0.5 s (4941),
# snapshots 60 120 180 240 s, hourly checkpoints. Log: log/2026-10-26_haringa54-state-limited-generator.md
# usage: CODE_DIR=... COUPLING=own|mean [WALLS="--smooth-walls --baffles sheets"] [RESUME=latest]
#        sbatch --export=ALL -J P54_own job_P54.sh P54_own
# WALLS (2026-10-29): generator options of the walls; "--smooth-walls --baffles sheets" = the 30 L production walls
# (log/2026-10-29_haringa54-production-walls.md); empty = the lattice cylinder and 90 mm solid baffles of P54.
NAME=${1:?NAME}
set -u
source /nesi/project/uoa04509/sph/env.sh
export FSI_ROOT=${CODE_DIR:?set CODE_DIR}
export PYTHONPATH=$FSI_ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export VK_DRIVER_FILES=/nesi/project/uoa04509/sph/nvidia_icd.json
cd $FSI_ROOT
OUT=/nesi/nobackup/uoa04509/sph/h54; CKPT=$OUT/${NAME}_ckpt; CASE=$OUT/${NAME}_case
mkdir -p $OUT
echo "host=$(hostname) job=$SLURM_JOB_ID $(date -Is) code=$FSI_ROOT name=$NAME coupling=$COUPLING walls=${WALLS:-} resume=${RESUME:-}"
nvidia-smi --query-gpu=index,name --format=csv,noheader
if [ -z "${RESUME:-}" ]; then
    rm -rf "$CASE" "$OUT/${NAME}_lifelines" "$OUT/${NAME}_snapshots" "$CKPT" "$OUT/${NAME}_probes.csv" "$OUT/${NAME}_torque.csv"
    python utils/geometry/_demo_rushton_tank.py --preset haringa54 --dx 0.030 --free-slip-lid --sgs --shaft-stop-at-lid \
        --substrate --state-limited --biomass 10 --q-max-umol-per-g-h 8300 --k-s 1e-5 --growth-yield 90 \
        --maintenance-umol-per-g-h 222 --demand-margin 0.1 --tau-up 20 --tau-down 20 --growth-rate-initial 6.94e-5 \
        --feed-rate 0.44 --feed-start 0 --state-coupling $COUPLING ${WALLS:-} --out "$CASE" | grep -E "fluid=|dt =|smooth|sheet"
    sed -i "s#material_library: materials.yaml#material_library: $CASE/materials.yaml#" "$CASE/case.yaml"
    RESUME_ARGS=""
else
    RESUME_ARGS="--resume $RESUME"
fi
EXTRA=""; [ "$COUPLING" = mean ] && EXTRA="--population-mean-state-every 0.5"
python /nesi/project/uoa04509/sph/run_check.py experiment/v1/_run_v1_headless.py "$CASE/case.yaml" --max-steps 2371800 \
    --probe-every 4941 --probe-log "$OUT/${NAME}_probes.csv" \
    --torque-every 2000 --torque-log "$OUT/${NAME}_torque.csv" --torque-split-height 2.4 --torque-shaft-radius 0.14 \
    --lifeline-dir "$OUT/${NAME}_lifelines" --lifeline-count 20000 --lifeline-start 0 --lifeline-every 988 \
    --lifeline-chunk-seconds 20 --lifeline-fields substrate growth_rate uptake \
    --scalar-snapshot-times 60 120 180 240 --scalar-snapshot-dir "$OUT/${NAME}_snapshots" --scalar-snapshot-fields substrate growth_rate uptake \
    --checkpoint-dir "$CKPT" --checkpoint-wall-minutes 60 $RESUME_ARGS $EXTRA --dump "$OUT/${NAME}_final.npz" \
    2>&1 | grep --line-buffered -vE "queue_family|^  \[|probes="
python experiment/v1/checks/_analyze_lifelines.py integrity "$OUT/${NAME}_lifelines" 2>&1 | tee "$OUT/${NAME}_lifeline_integrity.txt"
echo "done $(date -Is)"
