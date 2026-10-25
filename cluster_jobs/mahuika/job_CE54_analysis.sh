#!/bin/bash -l
#SBATCH -A uoa04509
#SBATCH -p genoa
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J CE54ana
#SBATCH -o /nesi/nobackup/uoa04509/sph/logs/CE54ana_%j.out
# C54 (local two-way) vs E54 (population-mean two-way), K_i 49.2, 54 m3, statistics 120..320 s (2026-10-25):
# _compare_knob_runs.py (probes, snapshots, lifeline pools and regimes), frozen-field test of both (0.1 m cells),
# and reduced lifeline packs (substrate only, from 120 s) of A54 / C54 / E54 for the local 80 h stitching.
set -u
source /nesi/project/uoa04509/sph/env.sh
export FSI_ROOT=${CODE_DIR:?set CODE_DIR}
export PYTHONPATH=$FSI_ROOT MPLBACKEND=Agg
cd $FSI_ROOT
H=/nesi/nobackup/uoa04509/sph/h54; OUT=$H/CE54_analysis; mkdir -p $OUT
for n in C54 E54; do d=$OUT/run_$n; mkdir -p $d
  ln -sfn $H/${n}_probes.csv $d/probes.csv; ln -sfn $H/${n}_lifelines $d/lifelines; ln -sfn $H/${n}_snapshots $d/snapshots
done
echo "=== compare $(date -Is)"
python experiment/v1/checks/_compare_knob_runs.py --run C=$OUT/run_C54 --run E=$OUT/run_E54 --ki 49.2 --start 120 --out $OUT/compare > $OUT/compare.txt 2>&1
for n in C54 E54; do
  echo "=== frozen $n $(date -Is)"
  python experiment/v1/checks/_analyze_lifelines.py frozen $H/${n}_lifelines --start 120 --cell 0.1 --min-samples 20 --monod 1 9.8e-6 > $OUT/frozen_$n.txt 2>&1
done
for n in A54 C54 E54; do
  echo "=== reduce $n $(date -Is)"
  python /nesi/project/uoa04509/sph/reduce_lifelines.py $H/${n}_lifelines $OUT/reduced/${n}_lifelines 120 substrate > $OUT/reduce_$n.txt 2>&1
done
echo "done $(date -Is)"
