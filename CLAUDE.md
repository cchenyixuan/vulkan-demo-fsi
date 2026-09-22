# Vulkan SPH — single-GPU boundary-condition / FSI branch

## Overview

Branch `v1-single-gpu-fsi`, forked from `v4-multigpu-orchestration` at commit
`0cf9487` (2026-09-22). It is a **single-GPU development line**: the goal here is
to extend the V1 δ-plus WCSPH solver core with additional boundary conditions
(inlet / outlet, moving walls, ...) and fluid–structure interaction (FSI).

Multi-GPU (ghost exchange, migration, transport backends, partitioning,
scaling campaigns, paper material) is **out of scope on this branch** and was
removed from the tree; it lives on `v4-multigpu-orchestration`. Do not
re-import it here.

Solver summary: δ-plus WCSPH, explicit leapfrog (half-step velocity storage,
5 kernels per step), persistent uniform voxel grid for neighbour search, Vulkan
compute via `python-vulkan`, case parameters delivered to shaders as
specialization constants.

## Current state

- **Core verified after the cut (2026-09-22)**: cavity 1M
  (`cases/lid_driven_cavity_2d`, 1,046,529 particles) runs headless at ~467
  steps/s on one RTX 5090, alive count conserved, no overflow. Final positions
  after 1200 steps agree with the pre-cut code to within the solver's own
  run-to-run GPU nondeterminism (atomic append order in `predict` /
  `update_voxel`; max |Δx| ≈ 4e-3 either way, mean ≈ 5e-5). The GLFW viewer
  (`experiment/v1/_run_v1_viewer.py`) also runs.
- **What the cut removed from the V1 code** (commit 2 of this branch): the
  `GhostTransportConfig` / ghost pool constructor arguments, `ghost_send` and
  `install_migrations` pipelines and shaders, the pre/post-sync split command
  buffers, the cross-GPU transport handle export, and the dual-GPU drivers,
  transport / partition modules and their run scripts.
- **What still remains from the V1 multi-GPU layout** (harmless, all pinned to
  zero): `common.glsl` / `helpers.glsl` still declare the ghost pool / ghost
  voxel spec constants (ids 54, 55, 80, 81) and the `own_first_pid()` /
  `is_own_voxel()` helpers; `simulator_v1.py` pins those constants to 0 so the
  pid layout is `[1, POOL_SIZE]` and the voxel layout `[1, NX*NY*NZ]`, and every
  ghost branch in the kernels is dead-code-eliminated. Descriptor set 2 is an
  empty placeholder and `GlobalStatusBuffer` keeps its 16-uint layout (the
  ghost / migration fields are always 0). When BC / FSI work touches
  `common.glsl` or `helpers.glsl`, these remnants may be deleted; nothing on
  this branch depends on them (then also drop the `SPEC_ID_*_GHOST_*` entries in
  `simulator_v1.py`).

## Layout

```
vulkan-demo-fsi/
├── CLAUDE.md
├── experiment/v1/
│   ├── README.md                 # V1 core orientation
│   ├── _run_v1_viewer.py         # entry point: GLFW viewer, one GPU
│   ├── _run_v1_headless.py       # entry point: N steps, no window (bench / regression dump)
│   ├── compile_shaders_v1.py     # glslc batch compile -> shaders/spv/ (+ spv/render/)
│   ├── shaders/                  # compute kernels + render shaders; README.md inside
│   │   ├── common.glsl           # spec constants + descriptor bindings (single source of truth)
│   │   ├── helpers.glsl          # kernel functions, voxel coord math
│   │   ├── initialize_voxelization.comp, predict.comp, update_voxel.comp,
│   │   │   correction.comp, density.comp, force.comp, bootstrap_half_kick.comp, defrag.comp
│   │   └── render/particle.{vert,frag}
│   └── utils/
│       ├── simulator_v1.py       # buffers, descriptors, pipelines, bootstrap/step/defrag cmds
│       └── renderer_v1.py        # point-sprite renderer, color modes, camera hotkeys
├── utils/sph/                    # shared infrastructure imported by V1
│   ├── case.py                   # case.yaml + material library loading, derived constants
│   ├── grid.py                   # voxel grid math (must match predict.comp)
│   ├── obj_loader.py             # particle .obj loading
│   ├── vulkan_context.py         # instance / device / queue / command pool
│   └── camera.py
├── utils/geometry/               # particle discretization tool (lattice / SDF / mesh / relax)
├── materials/standard.yaml       # material library (kind: fluid | boundary | inlet | rotor)
├── cases/lid_driven_cavity_2d/   # 1M 2D cavity, the default case
├── cases/cavity3d_1m/            # 1.03M 3D cavity
└── docs/sph_design.md, sph_v0_design.md, sph_v1_design.md
```

`*.obj` particle files and `*.spv` are gitignored. The case `.obj` files must be
regenerated (`utils/geometry/_demo_cavity_case.py`, `_demo_cavity_case_3d.py`)
or copied from another checkout; shaders are compiled automatically by the
run scripts.

## Code Conventions

- **No abbreviations in identifiers.** Use full words in variables, buffers, struct fields, and function names. Examples:
  - `correction_matrix` / `correction_matrix_inverse`, not `M` / `M_inv`
  - `neighbor_particle_id`, not `pid_j`
  - `smoothing_length`, not `h`
  - `kernel_gradient`, not `gW`
  - `PositionVoxelIdBuffer`, not `PosVidBuf`
  - Loop counters: `row_index` / `slot_index`, not `i` / `k`
- Math symbols (W, ρ, ∇, M, ξ) are **allowed in comments** explaining derivation; **never in identifiers**.
- Verbose names acceptable even when long: `smoothing_length_power_dimension_plus_one` beats `h_dim_p1`.
- `#include "common.glsl"` then `#include "helpers.glsl"` at the top of every `.comp`.

## Solver facts to know before changing anything

- **1-based ids everywhere**; slot 0 of every particle / voxel buffer is the
  dead / unallocated sentinel. `particle_id ∈ [1, POOL_SIZE]`,
  `voxel_id ∈ [1, NX*NY*NZ]`.
- **Per-step command buffer**: `predict → update_voxel → correction → density →
  (scratch→primary density copy) → force`. Bootstrap: `initialize_voxelization →
  correction → density → force → bootstrap_half_kick` (backward half-kick
  `v_{-1/2} = v_0 − ½ a_0 dt`), then one defrag.
- **Defrag** (`defrag.comp`, copy-back through scratch set 4) runs every
  `numerics.defrag_cadence` steps and re-packs the pool in voxel order. It only
  walks `inside_particle_index`, which excludes INLET particles: **any INLET
  particles in the pool are lost at the first defrag.** Inlet / outlet work
  must redesign this path.
- **Material kinds** FLUID / BOUNDARY / INLET / ROTOR come from
  `materials/standard.yaml`. `predict` moves only FLUID; BOUNDARY particles are
  static and a moving wall is expressed purely through `initial_velocity`
  (the cavity lid: `kind=boundary`, `initial_velocity=[1,0,0]`, positions never
  change). Moving boundaries / FSI bodies will need `predict` (or a new kernel)
  to advance non-fluid kinds and to re-voxelize them.
- **Per-kernel kind table and buffer access matrix** are in
  `experiment/v1/shaders/README.md`.
- **Spec constants**: ids and ranges are listed in `common.glsl` and mirrored by
  `_global_spec_entries()` in `simulator_v1.py`; both must be edited together.
  Free ranges for new constants: 34–39, 56–79, 89+.
- **Uniform-material simplifications** inherited from V0: `force.comp` uses
  self's mass / viscosity / volume for pair quantities (no multi-phase), no
  micropolar terms, no rotor motion, no inlet spawn kernel.
- **Overflow counters** in `GlobalStatusBuffer` (`readback_global_status()`)
  are the first thing to check after any change: `overflow_inside_count`,
  `overflow_incoming_count`, `correction_fallback_count` must stay 0.

## Environment

- Python 3.13 venv: this worktree has **no `.venv` of its own**; use the main
  checkout's interpreter `C:/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe`
  (point PyCharm at it) or create a new venv with `vulkan`, `glfw`, `numpy`,
  `pyrr`, `pyyaml`.
- Vulkan SDK at `C:/VulkanSDK/1.4.341.1/` (`glslc.exe`); override with `VULKAN_SDK`.
- Hardware (2026-09): 2× RTX 5090; this branch uses one. `device[0]` also
  drives the Windows desktop; pass `--device 1` to use the other card.
- Validation layers are **off by default** in both run scripts (they cost real
  fps with the full SDK installed); pass `--validation` while debugging.

## Running

```bash
PY=C:/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe

# Viewer (compiles shaders first): SPACE pause, 0..5 color modes, ESC quit
$PY experiment/v1/_run_v1_viewer.py                      # cavity 1M
$PY experiment/v1/_run_v1_viewer.py cases/cavity3d_1m/case.yaml --point-size 4

# Headless: N steps, prints alive / overflow / steps per second
$PY experiment/v1/_run_v1_headless.py --max-steps 2000
$PY experiment/v1/_run_v1_headless.py --max-steps 1200 --dump before.npz   # regression dump

# Compile shaders only
$PY experiment/v1/compile_shaders_v1.py
```

Regression check when changing kernels or the simulator: dump positions before
and after (`--dump`), compare alive counts exactly, and compare the sorted point
sets against the run-to-run noise of the unchanged build (two dumps of the same
build). The solver is not bit-reproducible run to run because voxel incoming
lists are filled by atomics.
