#!/bin/bash -l
#SBATCH -A uoa04509
#SBATCH -p genoa
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J A54ana
#SBATCH -o /nesi/nobackup/uoa04509/sph/logs/A54ana_%j.out
# A54 lifeline analysis (2026-10-18), CPU only: regime statistics and the frozen-field test from 120 s (statistics
# window 120..320 s) at two cell sizes, and the q spectrum. Output /nesi/nobackup/uoa04509/sph/h54/A54_analysis/.
set -u
source /nesi/project/uoa04509/sph/env.sh
export FSI_ROOT=${CODE_DIR:?set CODE_DIR}
export PYTHONPATH=$FSI_ROOT MPLBACKEND=Agg
cd $FSI_ROOT
L=/nesi/nobackup/uoa04509/sph/h54/A54_lifelines; OUT=/nesi/nobackup/uoa04509/sph/h54/A54_analysis; mkdir -p $OUT
A=experiment/v1/checks/_analyze_lifelines.py
for cell in 0.1 0.2; do
  echo "=== frozen cell $cell $(date -Is)"
  python $A frozen $L --start 120 --cell $cell --min-samples 20 --monod 1 9.8e-6 > $OUT/frozen_$cell.txt 2>&1
done
echo "=== spectrum $(date -Is)"
python $A spectrum $L --start 120 --monod 1 9.8e-6 --out $OUT/spectrum_q > $OUT/spectrum_q.txt 2>&1
echo "done $(date -Is)"
