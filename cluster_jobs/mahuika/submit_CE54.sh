#!/bin/bash
# C54 (local two-way, knob own) and E54 (two-way, knob with the population-mean X_gly every 0.05 s), K_i 49.2 = 2 X_gly,ss,
# k_E11 rescaled at X_gly,0 = 24.6 (the 30 L Da1 point at 54 m3). Mahuika, 2026-10-15.
cd /nesi/project/uoa04509/sph
export CODE_DIR=/nesi/project/uoa04509/sph/vulkan-demo-fsi_${CODE_COMMIT:-63be0f7}
STEADY="gly=24.6,aa=931,sto=2652,paa=2.71,e11=0.146,e32=211.9,e4=0.048,v33=4.06e-4,Cs=9.66e-6,CPAA=2.17e-3"
BASE="--free-slip-lid --sgs --shaft-stop-at-lid --ninepool --biomass 55 --ninepool-initial $STEADY --feed-rate 0.37 --feed-start 0"
export LIFELINE_FIELDS="substrate gly growth_rate sto uptake" SNAPSHOT_FIELDS="substrate gly growth_rate uptake"
GENFLAGS="$BASE --ninepool-ki 49.2" EXTRA="" sbatch --export=ALL -J C54 job_54.sh C54
GENFLAGS="$BASE --ninepool-ki 49.2 --ninepool-ki-mean" EXTRA="--population-mean-gly-every 0.05" sbatch --export=ALL -J E54 job_54.sh E54
