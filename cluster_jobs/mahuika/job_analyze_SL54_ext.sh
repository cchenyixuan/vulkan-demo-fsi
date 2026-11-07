#!/bin/bash -l
#SBATCH -A uoa04509
#SBATCH -p genoa
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o /nesi/nobackup/uoa04509/sph/logs/analyze_%x_%j.out
# CPU post-processing of a 54 m3 state-limited own / mean pair over the extension window (2026-11-08): the recipe of
# job_analyze_SL54.sh (0.1 m cells, K_s 1e-5, regime thresholds 0.05 / 0.95 with hysteresis 0.01 on the 0.36 s average) with the
# statistics window START..END instead of 120..240 s, plus the 20 s sub-windows (windows_runs2.py) and the substrate budget.
# Reads the md5-checked copies in /nesi/project/uoa04509/sph/production (read only).
# usage: TAU=45 START=240 END=480 sbatch --export=ALL -J SLt45ext job_analyze_SL54_ext.sh G24_SL_t45
PAIR=$1
set -u
source /nesi/project/uoa04509/sph/env.sh
export FSI_ROOT=/nesi/project/uoa04509/sph/vulkan-demo-fsi_688e87a
export PYTHONPATH=$FSI_ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $FSI_ROOT
P=/nesi/project/uoa04509/sph/production; T=/nesi/project/uoa04509/sph/analysis_tools; C=experiment/v1/checks
TAG=${PAIR}_${START}_${END}
A=/nesi/nobackup/uoa04509/sph/analysis/$TAG; KEEP=/nesi/project/uoa04509/sph/analysis/$TAG; mkdir -p $A $KEEP
stamp() { echo "$(date -Is) $1" | tee -a $A/progress.txt; }
stamp "start job $SLURM_JOB_ID pair $PAIR tau $TAU window $START..$END"
stamp "windows"
python $T/windows_runs2.py --start $START --end $END --ks 1e-5 $P/${PAIR}_own_lifelines $P/${PAIR}_mean_lifelines > $A/windows.txt 2>&1
echo "exit $?" >> $A/progress.txt
stamp "halves"; START=$START END=$END python $T/halves_runs2.py $P ${PAIR}_own ${PAIR}_mean > $A/halves.txt 2>&1; echo "exit $?" >> $A/progress.txt
stamp "fields"; START=$START python $T/fields_runs2.py $P ${PAIR}_own ${PAIR}_mean > $A/fields.txt 2>&1; echo "exit $?" >> $A/progress.txt
stamp "budget"; python $T/budget_runs.py $P ${PAIR}_own ${PAIR}_mean > $A/budget.txt 2>&1; echo "exit $?" >> $A/progress.txt
cp $A/*.txt $KEEP/
for n in own mean; do
  stamp "frozen $n"
  python $C/_analyze_lifelines.py frozen $P/${PAIR}_${n}_lifelines --start $START --cell 0.1 --min-samples 20 --monod 1 1e-5 \
      --out $A/frozen_$n > $A/frozen_$n.txt 2>&1; echo "exit $?" >> $A/progress.txt
done
stamp "own check"
python $C/_integrate_state_limited_lifelines.py $P/${PAIR}_own_lifelines --out $A/offline_own_check.npz --init-from-record --check \
    --tau-up $TAU --tau-down $TAU --statistics-start $START > $A/offline_own_check.txt 2>&1; echo "exit $?" >> $A/progress.txt
stamp "mean check"
python $C/_integrate_state_limited_lifelines.py $P/${PAIR}_mean_lifelines --out $A/offline_mean_check.npz --demand-state recorded-mean \
    --init-from-record --check --tau-up $TAU --tau-down $TAU --statistics-start $START > $A/offline_mean_check.txt 2>&1; echo "exit $?" >> $A/progress.txt
stamp "mean one-way"
python $C/_integrate_state_limited_lifelines.py $P/${PAIR}_mean_lifelines --out $A/offline_mean_oneway.npz --init-from-record \
    --tau-up $TAU --tau-down $TAU --statistics-start $START > $A/offline_mean_oneway.txt 2>&1; echo "exit $?" >> $A/progress.txt
for n in own mean; do
  stamp "layout $n"
  python $C/_analyze_regime_layout_54m3.py $P/${PAIR}_${n}_lifelines --k-s 1e-5 --start $START --out $A/layout_$n > $A/layout_$n.txt 2>&1
  echo "exit $?" >> $A/progress.txt
done
cp $A/*.txt $KEEP/
stamp "all done"
