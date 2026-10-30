#!/bin/bash
# Fast-adaptation limit of the state-limited model (2026-10-30, user: "跑这个对照算例，提交到paratera"): 30 L, 4 mm production
# config as P1 / the tau scan, one-way Monod sink at the nominal capacity q_max = 8300 umol/gdw/h, K_s 1e-5 mol/kg, x_bio 10 g/kg,
# no growth, feed 2.5e-4 mol/s from t = 0, 120 s, statistics 40-120 s. For tau -> 0 the state-limited cell takes up the full Monod
# supply, so this run is the limit the tau = 0.5 / 3 / 30 s pairs are compared with. Log: log/2026-10-30_monod-limit-30L-submission.md
cd ~/run/sph
export CODE_DIR=$HOME/run/sph/vulkan-demo-fsi_${CODE_COMMIT:?set CODE_COMMIT}
PROD="--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2 --sgs"
BIO="--substrate --biomass 10 --q-max-umol-per-g-h 8300 --k-s 1e-5 --feed-rate 2.5e-4 --feed-start 0"
A30="0.004 1371120 5713 10000 0 343 40 80 120"
export LIFELINE_FIELDS="substrate uptake" SNAPSHOT_FIELDS="substrate uptake"
GENFLAGS="$PROD $BIO" EXTRA="" sbatch --export=ALL -p gpu_5090,gpu_4090 --exclude=wqd10nbj04g3 -c 6 -t 08:00:00 -J P1_monod job_knob_pb.sh P1_monod $A30
squeue -u $USER
