# SPH V1 core — single GPU

The V1 δ-plus WCSPH solver core, reduced to one GPU. This directory is the
whole solver on the `v1-single-gpu-fsi` branch; boundary-condition and FSI
work happens here.

History: V1 started as the dual-GPU rewrite of the frozen V0 single-GPU code
(`utils/sph/simulator.py` + `shaders/sph/` on the main branches). On this
branch the dual-GPU drivers, transport / partition modules and the
`ghost_send` / `install_migrations` kernels were removed and
`SphSimulatorV1` was reduced to the single-GPU path; V0 itself was dropped so
there is exactly one simulator. Design lineage: `docs/sph_v0_design.md`
(buffer layout + pipeline), `docs/sph_v1_design.md` (the V1 merged-buffer
layout whose ghost parts are now pinned to zero), `docs/sph_design.md`
(method choices).

## Import rules

V1 imports shared infrastructure from `utils/sph/`:
- `utils.sph.case` — case YAML + material library loading, derived constants
- `utils.sph.grid` — voxel grid math
- `utils.sph.obj_loader` — particle geometry loading
- `utils.sph.vulkan_context` — instance / device / queues / command pool
- `utils.sph.camera` — viewer camera

Those modules are plain dependencies now (no frozen-V0 rule any more); edit
them when the solver needs it, keeping `case.py`'s constants in lockstep with
`shaders/common.glsl`.

## Layout

```
experiment/v1/
├── README.md                    # this file
├── _run_v1_viewer.py            # GLFW viewer entry point (compiles shaders first)
├── _run_v1_headless.py          # N steps without a window; bench + regression dump
├── compile_shaders_v1.py        # glslc: shaders/*.comp -> shaders/spv/, render/ -> spv/render/
├── shaders/                     # see shaders/README.md for the kernel-level orientation
│   ├── common.glsl              # spec constants + descriptor bindings
│   ├── helpers.glsl             # Wendland C4 kernel, voxel coord math, pid helpers
│   ├── initialize_voxelization.comp
│   ├── bootstrap_half_kick.comp
│   ├── predict.comp / update_voxel.comp / correction.comp / density.comp / force.comp
│   ├── defrag.comp
│   ├── _test_common.comp        # compile smoke test for common.glsl
│   └── render/particle.vert, particle.frag
└── utils/
    ├── simulator_v1.py          # SphSimulatorV1: buffers, descriptors, pipelines, cmd buffers
    └── renderer_v1.py           # SphRendererV1: swapchain, point sprites, color modes, hotkeys
```

## Simulator API

```python
with VulkanContext.create(application_name="...", enable_validation=False) as ctx:
    sim = SphSimulatorV1(ctx, case)      # allocates + uploads + records cmd buffers
    sim.bootstrap()                      # initial voxelization, a_0, v_{-1/2}, one defrag
    sim.step()                           # one leapfrog step (+ defrag on cadence)
    sim.run_until(max_steps=..., total_time=...)
    sim.readback_global_status()         # alive count + overflow counters (must be 0)
    sim.readback_positions()             # (POOL_SIZE + 1, 4) float32: x, y, z, voxel_id
    sim.get_render_buffers()             # handles the renderer binds
    sim.destroy()
```

`own_first_pid()` / `own_last_pid()` return `1` / `POOL_SIZE`; they are kept
because the renderer and `helpers.glsl` use them to describe the drawn / owned
pid range.
