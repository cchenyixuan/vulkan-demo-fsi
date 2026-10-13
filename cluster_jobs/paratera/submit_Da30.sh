#!/bin/bash
# 30 L Da scan on Paratera (2026-10-14, approved by the user: "先跑30 L Da 扫描吧，提交到并行"): local two-way (C, knob
# with the particle's own X_gly) against population-mean two-way (E, knob with the tank-mean X_gly every 0.05 s) at
# three points of Da_mix = tau_circ / t_uptake, t_uptake = C_s,bulk / (q_s X). Geometry and resolution fixed (30 L,
# 4 mm production config, 577,837 particles, dt 8.752e-5 s); Da raised through the biomass with the feed scaled in
# proportion so q / q_max stays the same; one extra point with K_i halved (feedback strength axis).
# Protocol: Tang 9-pool, pools at the 0-D chemostat steady state D = 0.05 (as B1 and the 54 m3 batch), continuous
# feed from t = 0 at the top injection point, 200 s = 2,285,200 steps, statistics 60..200 s (about 4.7 tau95),
# 10,000 lifelines from t = 0 every 0.03 s (343 steps), probes every 0.5 s (5713 steps), snapshots 60 100 140 200 s.
#   Da1_C / Da1_E   x_bio 55 g/kg,  feed 2.0e-4 mol/s, K_i 49.2 (2 X_ss)   Da about 1.5, the 54 m3 C/E low end
#   Da1k_C / Da1k_E x_bio 55,       feed 2.0e-4,       K_i 24.6 (1 X_ss)   the 54 m3 D low end
#   Da6_C / Da6_E   x_bio 220,      feed 8.0e-4,       K_i 49.2            Da about 6, close to the 54 m3 value 7
# B1 reference: C 5.5 h, E 6.2 h on a 5090 at 4 mm 200 s (5,000 lifelines every 0.1 s). Three jobs on
# gpu_5090/gpu_4090 at first; the user then asked for hp_5090 only (2026-10-14), so all six go there with -A hp5090.
cd ~/run/sph
export CODE_DIR=$HOME/run/sph/vulkan-demo-fsi_${CODE_COMMIT:?set CODE_COMMIT, e.g. d27b0e4}
EXCL="--exclude=wqd10nbj04g3"
PROD="--skin 0 --solid-pressure accumulate --solid-reaction-force --pair-correction reverse --thin-plates --background-pressure 2000 --smooth-walls --conformal-baffles --thin-layers 2 --sgs"
STEADY="gly=24.6,aa=931,sto=2652,paa=2.71,e11=0.146,e32=211.9,e4=0.048,v33=4.06e-4,Cs=9.66e-6,CPAA=2.17e-3"
ARGS="0.004 2285200 5713 10000 0 343 60 100 140 200"
export LIFELINE_FIELDS="substrate gly growth_rate sto uptake" SNAPSHOT_FIELDS="substrate gly growth_rate uptake"
submit() {   # name biomass feed ki mode(own|mean) partition-args...
    local name=$1 bio=$2 feed=$3 ki=$4 mode=$5; shift 5
    export GENFLAGS="$PROD --ninepool --biomass $bio --ninepool-initial $STEADY --feed-rate $feed --feed-start 0 --ninepool-ki $ki"
    export EXTRA=""
    if [ "$mode" = mean ]; then GENFLAGS="$GENFLAGS --ninepool-ki-mean"; EXTRA="--population-mean-gly-every 0.05"; fi
    sbatch --export=ALL "$@" $EXCL -c 6 -t 14:00:00 -J $name job_knob_pb.sh $name $ARGS
}
G="-A hp5090 -p hp_5090"; H="-A hp5090 -p hp_5090"
submit Da1_C  55  2.0e-4 49.2 own  $G
submit Da1_E  55  2.0e-4 49.2 mean $H
submit Da1k_C 55  2.0e-4 24.6 own  $G
submit Da1k_E 55  2.0e-4 24.6 mean $H
submit Da6_C  220 8.0e-4 49.2 own  $G
submit Da6_E  220 8.0e-4 49.2 mean $H
