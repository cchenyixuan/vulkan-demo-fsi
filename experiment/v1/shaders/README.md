# SPH V1 Compute Shaders (single GPU)

Vulkan compute shader implementation of δ-plus WCSPH, single-GPU. See
`docs/sph_v0_design.md` (buffer layout + pipeline), `docs/sph_v1_design.md`
(the V1 merged-buffer layout — its ghost parts are pinned to zero on this
branch) and `docs/sph_design.md` (method choices) for the full design.

This README is a quick orientation for anyone (including future Claude
sessions) opening this directory. Authoritative details live in the design
docs and in `common.glsl`.

---

## Pipeline (Leapfrog, 5 main stages + bootstrap)

```
Bootstrap (once at t = 0)
    ┌────────────────────────────────┐
    │ Python: voxelize x_0 → buffers │
    │ correction → density → force   │  produces a_0, ρ_0, P_0, shift_0, M_0⁻¹
    │ bootstrap_half_kick            │  v_0 → v_{-1/2}
    └────────────────────────────────┘
                 │
                 ▼
Main loop (uniform, every step):
    1. predict        kick (v_{n-1/2} → v_{n+1/2}) + drift + voxel-crossing detection
    2. update_voxel   compact inside_particle_index + merge incoming events
    3. correction     KCG matrix M⁻¹ + ∇ρ + kernel_sum
    4. density        continuity (ρ_{n+1}) + Tait EOS (P_{n+1})
    5. force          pressure (TIC) + viscosity (Morris) + gravity + PST shift
                          → a_{n+1}, shift_{n+1}
```

Stage order matters. density.comp writes the new ρ/P to the transient
`density_pressure_scratch`; the simulator immediately copies that back to
the canonical `density_pressure` (set 0 binding 1) inside the same step
cmd, so force.comp sees the freshly-written values. The next step's
predict consumes force's outputs (`acceleration`, `shift`).

Inlet / outlet kernels are not implemented yet (that is this branch's
boundary-condition work). `defrag.comp` is implemented (separate path from
the per-step main loop, triggered every `defrag_cadence` steps — see file
inventory below). `initialize_voxelization.comp` fills the voxel lists on the
GPU at bootstrap.

---

## File inventory

| File | Purpose |
|---|---|
| `common.glsl` | All spec constants, descriptor set bindings, scalar constants (kind tags, sentinels). Single source of truth. |
| `helpers.glsl` | Shared device functions: Wendland C4 kernel + gradient, voxel coord conversions, symmetric mat3 unpack. |
| `initialize_voxelization.comp` | Bootstrap — GPU-side initial voxelization (fills `inside_particle_count` / `inside_particle_index`). |
| `bootstrap_half_kick.comp` | One-time backward half-kick at t=0 (`v_0 → v_{-1/2}`). |
| `predict.comp` | Stage 1 — kick + drift + crossing detection. |
| `update_voxel.comp` | Stage 2 — per-voxel compaction + incoming merge. |
| `correction.comp` | Stage 3 — KCG correction matrix + density gradient + kernel sum. |
| `density.comp` | Stage 4 — continuity equation + EOS. |
| `force.comp` | Stage 5 — pressure + viscosity + gravity + PST shift (+ vorticity ω_z into `acceleration.w` for the viewer). |
| `defrag.comp` | Periodic — re-packs the pool in voxel order into scratch set 4; the simulator copies it back. |
| `_test_common.comp` | Smoke test: empty kernel including `common.glsl`, used to detect regressions in bindings/spec constants. |
| `render/particle.vert`, `render/particle.frag` | Point-sprite render shaders for `SphRendererV1` (color modes incl. vorticity). |

Compiled `.spv` artifacts go to `spv/` (gitignored). Build via
`experiment/v1/compile_shaders_v1.py` (the run scripts call it automatically).
Underscore-prefixed `.comp` files are skipped.

---

## Conventions

**1-based indexing throughout the SPH pipeline.** Every id is in `[1, N]`,
with `0` reserved as the universal "unallocated / dead / empty" sentinel:

- `particle_id ∈ [1, POOL_SIZE]` — slot 0 of every particle buffer is unused
- `voxel_id ∈ [1, TOTAL_VOXEL_COUNT]` — slot 0 of every voxel buffer is unused
- `slot_entry ∈ [1, POOL_SIZE]` (when storing particle refs in voxel buffers)

Shader entry: `uint self_particle_id = gl_GlobalInvocationID.x + 1u;`
followed by `if (self_particle_id > POOL_SIZE) return;` for padding threads.

Per-voxel kernels (`update_voxel`, `defrag`) use `TOTAL_VOXEL_COUNT` in the
bound check.

**Voxel coords stay 0-based** (spatial coord `0..GRID_DIM-1`). The 1-based
voxel_id is encoded as `coord.x + coord.y·Dx + coord.z·Dx·Dy + 1` (the +1
makes 0 a valid sentinel). `helpers.glsl` provides `own_coord_of(vid)` and
`own_voxel_id_of(coord)` to bridge the two.

**Shared helpers go in `helpers.glsl`**. Per-shader local helpers (functions
only one shader uses) stay local to that .comp file.

**No abbreviations in identifiers**. `correction_matrix` not `M`,
`smoothing_length` not `h`, `neighbor_particle_id` not `pid_j`. Math
notation (W, ρ, ∇, ξ) is fine in comments.

**`#include "common.glsl"` then `#include "helpers.glsl"`** at the top of
every `.comp` file. Both are header-guarded.

---

## Material / kind handling per kernel

| Kernel | FLUID | BOUNDARY | ROTOR | INLET | DEAD (vid=0) |
|---|---|---|---|---|---|
| `bootstrap_half_kick` | run | skip | skip | skip | skip |
| `predict` | run | skip | rigid-body update (prescribed rotation, 2026-09-25) | skip | skip |
| `update_voxel` | per-voxel kernel; not per-particle | | | | |
| `correction` | run | run | run | run* | skip |
| `density` | run | run (stores ρ₀) | run (stores ρ₀, as BOUNDARY) | skip | skip |
| `force` | run | run | run | skip | skip |
| `defrag` | per-voxel kernel; walks `inside_particle_index` only (INLET particles are not preserved — see below) | | | | |

\* `correction` does not skip INLET because doing so would require pulling in
`material[pid] + material_parameters[group]` (~52 B/particle) just for the
kind check — for typical inlet population (< 1% of pool) this costs more than
it saves. INLET particles' correction outputs are written but unread (they
never appear as neighbors).

ROTOR particles compute the full force/density/correction in V0 even though
predict skips them (no motion). This is a deliberate carry-over from legacy;
rotor coupling will be revisited in V0+.

---

## Buffer access matrix

Reads (R) and Writes (W) per kernel. See `common.glsl` for binding numbers.

| Buffer (set 0 unless noted) | bootstrap | predict | update_voxel | correction | density | force |
|---|---|---|---|---|---|---|
| `position_voxel_id` | R | R/W | R/W (kill) | R | R | R |
| `density_pressure` | — | — | — | R | R | R |
| `density_pressure_scratch` | — | — | — | — | W | — |
| `velocity_mass` | R/W | R/W | R/W (kill) | R (mass via .w) | R | R |
| `acceleration` | R | R | — | — | — | W |
| `shift` | — | R | — | — | — | W |
| `material` (group_id) | R | R | — | — | R | R |
| `correction_inverse` | — | — | — | W | R | R (self) |
| `density_gradient_kernel_sum` | — | — | — | W | R | R (.w only, self) |
| `inside_particle_count` (set 1) | — | — | R/W | R | R | R |
| `inside_particle_index` (set 1) | — | — | R/W | R | R | R |
| `incoming_particle_count` (set 1) | — | atomic R/W | R/W (reset) | — | — | — |
| `incoming_particle_index` (set 1) | — | atomic W | R | — | — | — |
| `material_parameters` (set 3) | R (kind) | R (kind) | — | — | R (kind, eos, ρ_0) | R (kind, ν, R, V) |
| `global_status` (set 3, atomics) | — | atomic W on overflow | atomic W on overflow | atomic W on fallback | — | — |

Notes:
- Density staging is **scratch+copy**: density.comp writes
  `density_pressure_scratch` (binding 2); the simulator's step cmd then
  issues `vkCmdCopyBuffer scratch → primary` so by force.comp the
  canonical `density_pressure` (binding 1) holds ρ_{n+1}, P_{n+1}.
  No ping-pong, no descriptor parity.
- `correction` does not read `material` to skip INLET (see notes above).
- `force` is the most expensive: reads neighbor M, ∇ρ, mass, density, vel for
  pair-averaged formulas; uniform V0 lets us compute pair quantities from
  self only (see `force.comp` header for what's hoisted).

---

## Invariants (consumed across kernels)

These are written into `common.glsl`'s buffer comments where relevant.
Summarized here for orientation:

1. **Contiguous packing** in `inside_particle_index[v * CAP + slot]`: valid
   particle_ids occupy slots `[0, count)`; trailing slots are 0.
   Maintained by `update_voxel.comp`'s Phase 3.

2. **No inlet/dead in voxel buffers**. `inside_particle_index` and
   `incoming_particle_index` never contain inlet or dead particle_ids.
   - `predict.comp` never atomic-appends inlet (kind check) or dead
     (skipped early).
   - Initial Python-side voxelization must skip inlet particles when
     populating `inside_particle_index`.
   - `update_voxel.comp`'s Phase 1 filter (voxel_id mismatch) automatically
     drops dead particles (their voxel_id = 0 ≠ self).

3. **Incoming counter may exceed `MAX_INCOMING_PER_VOXEL`** when predict
   overflows. `update_voxel.comp` clamps via `min(count, CAP)` before
   reading.

4. **1-based id everywhere** with 0 = sentinel (see Conventions above).

5. **Ghost remnants are inert.** `common.glsl` / `helpers.glsl` still carry
   the V1 ghost pool / ghost voxel spec constants (ids 54, 55, 80, 81) and
   the `own_first_pid()` / `is_own_voxel()` helpers. `simulator_v1.py` pins
   the constants to 0, so `own_first_pid() = 1`, every voxel is "own", and
   the ghost branches are dead-code-eliminated. They may be removed once BC /
   FSI work touches these headers.

---

## Bootstrap convention (backward half-kick)

Predict's kick formula `v_{n+1/2} = v_{n-1/2} + a_n · dt` consumes the
acceleration left in the buffer by the previous step's force. For step 1 to
produce correct `v_{1/2}` from buffer-stored `a_0`, the buffer must hold
**`v_{-1/2}` (backward offset)**, computed as:

```
v_{-1/2} = v_0 - 0.5 · a_0 · dt
```

`bootstrap_half_kick.comp` does exactly this. Forward bootstrap
(`v_{1/2} = v_0 + 0.5 a_0 dt`) would feed step 1 a stale `a_0` instead of
fresh `a_1`, introducing a permanent O(dt²) trajectory error from step 1.

Full bootstrap sequence (run once before main loop):

1. Python: populate particle buffers from scene IC; `initialize_voxelization.comp`
   fills `inside_particle_index` (skip inlet).
2. `correction.comp` on x_0
3. `density.comp` on x_0 → ρ_0, P_0
4. `force.comp` on x_0, v_0, ρ_0 → a_0, shift_0
5. `bootstrap_half_kick.comp` → v_{-1/2}
6. Enter main loop at step 1.

---

## Compile

`python experiment/v1/compile_shaders_v1.py` (run from repo root) produces
`spv/<name>.comp.spv` for every `.comp` and `spv/render/particle.{vert,frag}.spv`
for the render shaders. Uses `glslc` from VULKAN_SDK with:

- `-O` optimization (dead-code-eliminates unused bindings declared in
  `common.glsl`)
- `-I experiment/v1/shaders` so `#include` resolves
- `--target-env=vulkan1.2` (SPIR-V 1.5; the VkInstance is Vulkan 1.2)

Underscore-prefixed `.comp` (currently only `_test_common.comp`) are skipped;
compile `_test_common.comp` by hand as a regression check after
`common.glsl` edits.

---

## Spec constant ID range plan (see `common.glsl`)

```
0  - 9   : core physics scalars + grid origin  (id=3 reserved)
10       : STRICT_BIT_EXACT (multi-GPU remnant, pinned 0)
11 - 13  : grid dimensions (and TOTAL_VOXEL_COUNT derived)
14 - 16  : correction regularization tunables
17 - 19  : gravity
20 - 29  : voxel layout / micropolar (reserved)
30 - 33  : dimension + kernel coefficients
34 - 39  : free
40 - 49  : SPH numerical parameters (ε_h², PST main, PST anti, toggles, ...)
50 - 53  : capacities + workgroup size + POOL_SIZE
54 - 55  : ghost pool sizes (multi-GPU remnant, pinned 0)
56 - 79  : free
80 - 81  : ghost voxel counts (multi-GPU remnant, pinned 0)
82 - 127 : free (90-94 were the removed ghost_send / install_migrations ids)
```

New constants for boundary conditions / FSI go into the free ranges; add the
matching `SPEC_ID_*` + `_global_spec_entries()` line in `simulator_v1.py`.

---

## V0 simplifications (to be revisited in V0+)

- **Globally uniform mass / viscosity / radius / volume**: force.comp does
  NOT read neighbor MaterialParameters (uses self's values for all
  pair-averaged quantities). To enable multi-phase, restore neighbor reads
  inside the loop and compute harmonic means — see `force.comp` header.

- **No multi-phase rest_density scaling** in density's ψ_ij: legacy form
  `ρ_j · (ρ_0i / ρ_0j) - ρ_i` simplified to `ρ_j - ρ_i`.

- **No micropolar terms** (a_transfer, aw_*, curl_u in legacy force).

- **No rotor analytical motion**: rotor particles run all kernels but
  don't actually move (predict skips them).

- **No inlet spawn kernel**: inlet templates exist as static particles in
  the pool; no new particles are emitted at runtime. ⚠ Defrag does not
  preserve inlet pool slots — `defrag.comp` only walks
  `inside_particle_index`, which by invariant excludes inlet kind. Any case
  that has INLET particles in the main pool will lose them after the first
  defrag. The current lid case has no INLET so this is dormant; V0+ inlet
  spawn must redesign the defrag path (e.g. dedicated inlet pool list, or
  rebuild inlet slots from `inlet_template` post-defrag).

- **Single GPU**: the ghost pool / ghost voxel spec constants are pinned to
  0 (see invariant 5), so all ghost-related branches are dead-code-eliminated.

- **Per-particle dispatch over POOL_SIZE**: inlet/dead particles are not
  excluded at the dispatch level. Each kernel does its own bounds + kind
  check. Adding indirect dispatch (with maintained `alive_count`) deferred
  until inlet/outlet logic is implemented.

- **No bit-exact integration**: `STRICT_BIT_EXACT` spec constant exists but
  is pinned to 0 (no `precise` qualifiers in main kernels); it only mattered
  for multi-GPU ghost integration.

---

## What's NOT here yet

- `update_dispatch.comp` — 1-thread helper to update DispatchIndirectBuffer.
  Skipped because all kernels dispatch over POOL_SIZE directly.
- `inlet_spawn.comp` — spawn new fluid particles from inlet templates.
- Moving / deformable boundaries and FSI coupling — this branch's work.
- Numerical validation against the legacy OpenGL solver — not done.
