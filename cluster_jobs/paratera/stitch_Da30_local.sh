#!/bin/bash
# local 80 h stitching of the Da-scan lifelines (reduced packs), six processes; mirrors ~/run/sph/stitch_Da30.sh on Paratera
cd "D:/Uni of Auckland Dropbox/Beichen Zhao/CFD Projects/SPH lifeline/vulkan-demo-fsi"
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTHONIOENCODING=utf-8
PY="D:/Coding/Anaconda3/python.exe"; L=output/Da30/lifelines; OUT=output/Da30/stitch
INIT="gly=15.5,aa=931,sto=2652,paa=2.71,e11=0.146,e32=211.9,e4=0.048,v33=4.06e-4"
for p in Da1:49.2:55 Da1k:24.6:55 Da6:49.2:220; do
  pt=${p%%:*}; rest=${p#*:}; ki=${rest%%:*}; bio=${rest##*:}
  for n in C E; do
    "$PY" experiment/v1/checks/_integrate_ninepool_lifelines.py $L/${pt}_${n}_lifelines --out $OUT/stitch80h_${pt}_${n}.npz \
      --xbio $bio --paa 2.17e-3 --init "$INIT" --ki $ki --ki-reference-gly 24.6 --freeze-e11 --start 60 --step 0.03 \
      --stitch-hours 80 --stitch-count 2500 --stitch-report 2 --stitch-seed 1 > $OUT/stitch80h_${pt}_${n}.log 2>&1 &
  done
done
wait
echo "all-done $(date -Is)" > $OUT/ALL_DONE
