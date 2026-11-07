#!/bin/bash -l
#SBATCH -A uoa04509
#SBATCH -p genoa
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:15:00
#SBATCH --qos=debug
#SBATCH -o /nesi/nobackup/uoa04509/sph/logs/analyze_%x_%j.out
# quick part of job_analyze_SL54_ext.sh: S whole and 20 s sub-windows of START..END for an own / mean pair (debug QoS, 2026-11-08)
# usage: START=240 END=480 sbatch --export=ALL -J SLt45win job_windows_ext.sh G24_SL_t45
PAIR=$1
set -u
source /nesi/project/uoa04509/sph/env.sh
export FSI_ROOT=/nesi/project/uoa04509/sph/vulkan-demo-fsi_688e87a
export PYTHONPATH=$FSI_ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $FSI_ROOT
P=/nesi/project/uoa04509/sph/production; T=/nesi/project/uoa04509/sph/analysis_tools
KEEP=/nesi/project/uoa04509/sph/analysis/${PAIR}_${START}_${END}; mkdir -p $KEEP
date -Is
python $T/windows_runs2.py --start $START --end $END --ks 1e-5 $P/${PAIR}_own_lifelines $P/${PAIR}_mean_lifelines | tee $KEEP/windows_quick.txt
echo "exit $?"; date -Is
