"""Reduced copy of a lifeline directory for local analysis: records from --start, substrate only, positions zeroed
(compressed away), one chunk, same file format as the recorder so load_lifelines() reads it unchanged."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment.v1.utils.lifeline_recorder import load_lifelines
src, dst, start = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
fields = sys.argv[4].split(",") if len(sys.argv) > 4 else ["substrate"]
d = load_lifelines(src, with_aux=False)
keep = d["time"] >= start
idx = [d["field_names"].index(f) for f in fields]
dst.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst / "lifeline_0000.npz", time=d["time"][keep], step=d["step"][keep], rotor_angle=d["rotor_angle"][keep],
                    uid=d["uid"], position=np.zeros((int(keep.sum()), d["uid"].size, 3), np.float16),
                    scalars=np.ascontiguousarray(d["scalars"][keep][:, :, idx]), field_names=np.array(fields))
meta = dict(d["meta"]); meta.update(field_names=fields, records=int(keep.sum()), chunks=1, reduced_from=str(src), start_time_s=float(d["time"][keep][0]))
(dst / "lifeline_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(dst, keep.sum(), "records", fields, round((dst / "lifeline_0000.npz").stat().st_size / 1e6), "MB")
