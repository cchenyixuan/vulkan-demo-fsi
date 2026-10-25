#!/bin/bash
# Case-1 pre-test on Paratera (2026-10-25, user: "把预扫描放到并行上，用两张卡跑"): 30 L, 4 mm, state_limited with
# E. coli-like demonstration parameters, tau_up = tau_down = 3 s, x_bio 10 g/kg, feed 2.5e-4 mol/s from t = 0, 120 s;
# P1_own = local two-way (demand from the particle's own mu), P1_mean = population-mean state written every 0.05 s.
# 10,000 lifelines every 0.03 s (343 steps), probes 0.5 s, snapshots 40 80 120 s. Log: log/2026-10-25_state-limited-population-mean.md
cd ~/run/sph
export CODE_DIR=$HOME/run/sph/vulkan-demo-fsi_${CODE_COMMIT:-7185da1}
EXCL="--exclude=wqd10nbj04g3"
PROD="--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2 --sgs"
BIO="--substrate --state-limited --biomass 10 --q-max-umol-per-g-h 8300 --k-s 1e-5 --growth-yield 90 --maintenance-umol-per-g-h 222 --demand-margin 0.1 --tau-up 3 --tau-down 3 --growth-rate-initial 6.94e-5 --feed-rate 2.5e-4 --feed-start 0"
ARGS="0.004 1371120 5713 10000 0 343 40 80 120"
export LIFELINE_FIELDS="substrate growth_rate uptake" SNAPSHOT_FIELDS="substrate growth_rate uptake"
GENFLAGS="$PROD $BIO --state-coupling own" EXTRA="" sbatch --export=ALL -p gpu_5090 $EXCL -c 6 -t 06:00:00 -J P1_own job_knob_pb.sh P1_own $ARGS
GENFLAGS="$PROD $BIO --state-coupling mean" EXTRA="--population-mean-state-every 0.05" sbatch --export=ALL -p gpu_5090 $EXCL -c 6 -t 06:00:00 -J P1_mean job_knob_pb.sh P1_mean $ARGS
