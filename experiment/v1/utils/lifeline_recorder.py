"""lifeline_recorder.py — trajectories of a fixed sample of FLUID particles (2026-10-01, stage 3).

The cells of the lifeline study ride on the FLUID particles (docs/stage3_lifeline_design_2026-10-01.md):
every fluid particle is a Lagrangian parcel, so a lifeline is the history of one particle, identified
by its persistent uid (set 0 binding 10, the upload slot, carried through every defrag). This module
picks a random sample of fluid uids once and, every `every` steps from `start_time` on, reads back
their positions and, optionally, scalar fields, velocities and particle shifts.

Output (DIR):
    lifeline_meta.json            sample size, seed, record interval, field names, case path
    lifeline_NNNN.npz             one chunk of consecutive records:
        time (n,), step (n,), rotor_angle (n,)          float64 / int64
        uid (K,)                                        uint32, the same order in every chunk
        position (n, K, 3)                              float32, NaN for a particle not found
        scalars (n, K, F)                               float32 (with `fields`), compensated sum
        field_names (F,)
        aux_index (m,), velocity (m, K, 3), shift (m, K, 3)   every `aux_every`-th record
                                                        (with `aux`): stored velocity v_{n+1/2}
                                                        and the shift of the last step, float32

Cost: one read back of the position and uid buffers per record (plus scalars / velocity / shift when
asked); 3 mm tank, 1.36 M particles: about 30 MB, tens of milliseconds, every 0.03 s of flow time.
"""
import json
import pathlib
from typing import Optional, Sequence

import numpy as np


class LifelineRecorder:
    def __init__(self, simulator, directory, count: int, every: int, start_time: float = 0.0,
                 seed: int = 1, chunk_records: int = 333, fields: Optional[Sequence[str]] = None,
                 aux: bool = False, aux_every: int = 10, release_sphere: Optional[Sequence[float]] = None):
        self.simulator = simulator
        self.case = simulator.case
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.count = int(count)
        self.every = int(every)
        self.start_time = float(start_time)
        self.seed = int(seed)
        self.chunk_records = max(1, int(chunk_records))
        self.aux = bool(aux)
        self.aux_every = max(1, int(aux_every))
        self.release_sphere = None if release_sphere is None else np.asarray(release_sphere, dtype=np.float64)
        self.field_names = list(fields or [])
        if self.field_names:
            if self.case.scalars is None:
                raise ValueError("lifeline fields need a case with a `scalars:` block")
            all_names = self.case.scalars.field_names
            unknown = [name for name in self.field_names if name not in all_names]
            if unknown:
                raise ValueError(f"lifeline fields: unknown field(s) {unknown}")
            self.field_index = [all_names.index(name) for name in self.field_names]
        self.uids = None
        self.chunk_index = 0
        self.record_count = 0
        self.lost_total = 0
        self._reset_chunk()

    # ------------------------------------------------------------------
    def _reset_chunk(self):
        self.buffer = {"time": [], "step": [], "rotor_angle": [], "position": [], "scalars": [],
                       "aux_index": [], "velocity": [], "shift": []}

    def due(self) -> bool:
        sim = self.simulator
        return sim.step_count % self.every == 0 and sim.simulation_time >= self.start_time - 0.5 * self.case.timestep

    def _select(self, positions, live):
        sim = self.simulator
        material = sim.readback_material()
        fluid = live & np.isin(material, np.asarray(sim.fluid_group_ids(), dtype=material.dtype))
        if self.release_sphere is not None:
            centre, radius = self.release_sphere[:3], float(self.release_sphere[3])
            fluid &= np.linalg.norm(positions[:, :3].astype(np.float64) - centre, axis=1) < radius
        candidates = np.flatnonzero(fluid)
        if candidates.size == 0:
            raise RuntimeError("lifeline sample: no fluid particle to choose from")
        uid = sim.readback_particle_uid()
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(candidates, size=min(self.count, candidates.size), replace=False)
        self.uids = np.sort(uid[chosen]).astype(np.uint32)
        self.count = int(self.uids.size)
        meta = {"count": self.count, "seed": self.seed, "every_steps": self.every,
                "record_interval_s": self.every * self.case.timestep, "timestep_s": self.case.timestep,
                "start_time_s": sim.simulation_time, "start_step": sim.step_count,
                "field_names": self.field_names, "aux": self.aux, "aux_every": self.aux_every,
                "release_sphere": None if self.release_sphere is None else self.release_sphere.tolist(),
                "fluid_candidates": int(candidates.size),
                "particle_spacing_m": 2.0 * float(self.case.physics.particle_radius),
                "smoothing_length_m": float(self.case.physics.h)}
        (self.directory / "lifeline_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[lifeline] sampled {self.count:,} of {candidates.size:,} fluid particles (seed {self.seed}), "
              f"every {self.every} steps = {meta['record_interval_s']:.4f} s, from t = {sim.simulation_time:.3f} s "
              f"-> {self.directory}")

    def record(self):
        sim = self.simulator
        positions = sim.readback_positions()
        live = sim.live_slot_mask(positions)
        if self.uids is None:
            self._select(positions, live)
        uid = sim.readback_particle_uid()
        lookup = np.full(uid.shape[0] + 1, -1, dtype=np.int64)
        live_slots = np.flatnonzero(live)
        valid = uid[live_slots] < lookup.shape[0]
        lookup[uid[live_slots[valid]]] = live_slots[valid]
        slots = lookup[np.minimum(self.uids, lookup.shape[0] - 1)]
        lost = slots < 0
        safe = np.where(lost, 0, slots)
        position = positions[safe, :3].astype(np.float32)
        position[lost] = np.nan
        self.lost_total += int(lost.sum())
        buffer = self.buffer
        buffer["time"].append(sim.simulation_time)
        buffer["step"].append(sim.step_count)
        buffer["rotor_angle"].append(float(getattr(sim, "rotor_angle", 0.0)))
        buffer["position"].append(position)
        if self.field_names:
            values = sim.readback_scalars()[safe][:, self.field_index].astype(np.float64)
            if self.case.scalars.compensated_sum:
                values -= sim.readback_scalar_compensation()[safe][:, self.field_index].astype(np.float64)
            values = values.astype(np.float32)
            values[lost] = np.nan
            buffer["scalars"].append(values)
        if self.aux and self.record_count % self.aux_every == 0:
            velocity = sim.readback_velocity_mass()[safe, :3].astype(np.float32)
            shift = sim.readback_shift()[safe, :3].astype(np.float32)
            velocity[lost] = np.nan
            shift[lost] = np.nan
            buffer["aux_index"].append(len(buffer["time"]) - 1)
            buffer["velocity"].append(velocity)
            buffer["shift"].append(shift)
        self.record_count += 1
        if len(buffer["time"]) >= self.chunk_records:
            self.flush()

    def flush(self):
        buffer = self.buffer
        if not buffer["time"]:
            return
        arrays = {"time": np.asarray(buffer["time"], dtype=np.float64),
                  "step": np.asarray(buffer["step"], dtype=np.int64),
                  "rotor_angle": np.asarray(buffer["rotor_angle"], dtype=np.float64),
                  "uid": self.uids, "position": np.stack(buffer["position"])}
        if self.field_names:
            arrays["scalars"] = np.stack(buffer["scalars"])
            arrays["field_names"] = np.asarray(self.field_names)
        if buffer["aux_index"]:
            arrays["aux_index"] = np.asarray(buffer["aux_index"], dtype=np.int64)
            arrays["velocity"] = np.stack(buffer["velocity"])
            arrays["shift"] = np.stack(buffer["shift"])
        path = self.directory / f"lifeline_{self.chunk_index:04d}.npz"
        np.savez(path, **arrays)
        print(f"[lifeline] chunk {self.chunk_index}: {arrays['time'].size} records, "
              f"t = {arrays['time'][0]:.3f} .. {arrays['time'][-1]:.3f} s, lost so far {self.lost_total} -> {path.name}",
              flush=True)
        self.chunk_index += 1
        self._reset_chunk()

    def close(self):
        self.flush()
        meta_path = self.directory / "lifeline_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({"records": self.record_count, "chunks": self.chunk_index, "lost_total": self.lost_total})
            meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_lifelines(directory, first_chunk: int = 0, last_chunk: Optional[int] = None,
                   with_aux: bool = True) -> dict:
    """Concatenate the chunks of a lifeline directory: time (n,), step, rotor_angle, uid (K,),
    position (n, K, 3), and scalars / field_names / aux_index / velocity / shift when present."""
    directory = pathlib.Path(directory)
    meta = json.loads((directory / "lifeline_meta.json").read_text(encoding="utf-8"))
    paths = sorted(directory.glob("lifeline_[0-9][0-9][0-9][0-9].npz"))
    paths = paths[first_chunk:None if last_chunk is None else last_chunk + 1]
    if not paths:
        raise FileNotFoundError(f"no lifeline chunks in {directory}")
    parts = {key: [] for key in ("time", "step", "rotor_angle", "position", "scalars", "aux_index", "velocity", "shift")}
    offset = 0
    uid = None
    field_names = None
    for path in paths:
        with np.load(path) as chunk:
            if uid is None:
                uid = chunk["uid"]
            elif not np.array_equal(uid, chunk["uid"]):
                raise ValueError(f"{path.name}: sample uids differ from the first chunk")
            for key in ("time", "step", "rotor_angle", "position"):
                parts[key].append(chunk[key])
            if "scalars" in chunk.files:
                parts["scalars"].append(chunk["scalars"])
                field_names = [str(name) for name in chunk["field_names"]]
            if with_aux and "aux_index" in chunk.files:
                parts["aux_index"].append(chunk["aux_index"] + offset)
                parts["velocity"].append(chunk["velocity"])
                parts["shift"].append(chunk["shift"])
            offset += chunk["time"].size
    data = {"meta": meta, "uid": uid}
    for key, values in parts.items():
        if values:
            data[key] = np.concatenate(values)
    if field_names is not None:
        data["field_names"] = field_names
    return data
