#!/bin/bash -l
#SBATCH -A uoa04509
#SBATCH -p genoa
#SBATCH --gres=gpu:pro_6000:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 4-00:00:00
#SBATCH -J A54
#SBATCH -o /nesi/nobackup/uoa04509/sph/logs/A54_%j.out
# 54 m3 Haringa tank (preset haringa54), one-way Monod sink + 8 tracer pulses (2026-10-15, user: "A54提交mahuika").
# 25 mm, 3,907,706 particles, dt 8.4325e-5 s, 320 s = 3,796,000 steps. Monod q_max 1354 umol/gdw/h (D = 0.05 chemostat
# capacity of the Tang 9-pool model), K_s 9.8e-6, x_bio 55 g/kg, glucose 0.37 mol/s from t = 0 in the top feed sphere,
# pools integrated offline along the lifelines. Tracers at 60..102 s every 6 s (Haringa 2023 mixing time, bottom probe).
# Lifelines 20,000 from t = 0 every 0.1 s (1186 steps), probes every 0.5 s (5930 steps), snapshots 60 120 180 240 320 s,
# torque every 2000 steps, hourly checkpoints (RESUME=latest continues).
# usage: CODE_DIR=/nesi/project/uoa04509/sph/vulkan-demo-fsi_<commit> [RESUME=latest] sbatch --export=ALL job_A54.sh
set -u
source /nesi/project/uoa04509/sph/env.sh
export FSI_ROOT=${CODE_DIR:?set CODE_DIR}
export PYTHONPATH=$FSI_ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg
[ -f /nesi/project/uoa04509/sph/nvidia_icd.json ] && export VK_DRIVER_FILES=/nesi/project/uoa04509/sph/nvidia_icd.json
cd $FSI_ROOT
NAME=A54; OUT=/nesi/nobackup/uoa04509/sph/h54; CKPT=$OUT/${NAME}_ckpt; CASE=$OUT/${NAME}_case
mkdir -p $OUT
echo "host=$(hostname) job=$SLURM_JOB_ID $(date -Is) code=$FSI_ROOT resume=${RESUME:-}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
if [ -z "${RESUME:-}" ]; then
    rm -rf "$CASE" "$OUT/${NAME}_lifelines" "$OUT/${NAME}_snapshots" "$CKPT" "$OUT/${NAME}_probes.csv" "$OUT/${NAME}_torque.csv"
    python utils/geometry/_demo_rushton_tank.py --preset haringa54 --dx 0.025 --free-slip-lid --sgs --shaft-stop-at-lid \
        --substrate --biomass 55 --q-max-umol-per-g-h 1354 --k-s 9.8e-6 --feed-rate 0.37 --feed-start 0 \
        --tracers 8 --injection-start 60 --injection-interval 6 --out "$CASE" | grep -E "fluid=|dt =|shaft"
    sed -i "s#material_library: materials.yaml#material_library: $CASE/materials.yaml#" "$CASE/case.yaml"
    RESUME_ARGS=""
else
    RESUME_ARGS="--resume $RESUME"
fi
python /nesi/project/uoa04509/sph/run_check.py experiment/v1/_run_v1_headless.py "$CASE/case.yaml" --max-steps 3796000 \
    --probe-every 5930 --probe-log "$OUT/${NAME}_probes.csv" \
    --torque-every 2000 --torque-log "$OUT/${NAME}_torque.csv" --torque-split-height 2.4 --torque-shaft-radius 0.14 \
    --lifeline-dir "$OUT/${NAME}_lifelines" --lifeline-count 20000 --lifeline-start 0 --lifeline-every 1186 \
    --lifeline-chunk-seconds 20 --lifeline-fields substrate uptake \
    --scalar-snapshot-times 60 120 180 240 320 --scalar-snapshot-dir "$OUT/${NAME}_snapshots" --scalar-snapshot-fields substrate uptake \
    --checkpoint-dir "$CKPT" --checkpoint-wall-minutes 60 $RESUME_ARGS --dump "$OUT/${NAME}_final.npz" \
    2>&1 | grep --line-buffered -vE "queue_family|^  \[|probes="
python experiment/v1/checks/_analyze_lifelines.py integrity "$OUT/${NAME}_lifelines" 2>&1 | tee "$OUT/${NAME}_lifeline_integrity.txt"
echo "done $(date -Is)"
