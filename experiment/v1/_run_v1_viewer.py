"""
_run_v1_viewer.py — Lightweight single-GPU Vulkan + GLFW SPH viewer (V1).

One GPU running the 5-kernel leapfrog step (predict → update_voxel →
correction → density → force) and one window rendering every particle.
This branch is single-GPU only; the multi-GPU stack lives on the
v4-multigpu-orchestration branch.

Two performance levers for a smooth demo:
  * validation is OFF by default (Vulkan validation layers cost real fps on a
    machine with the full SDK installed). Pass --validation to re-enable.
  * --point-size controls the rendered sprite radius (default 12.0). Fragment
    fill / overdraw of large point sprites is the rendering bottleneck (see
    CLAUDE.md perf notes), so a smaller size renders much cheaper: drop to 3-5
    for max fps, raise for a fuller look. Adjust live with the ; / ' keys.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v1/_run_v1_viewer.py [case_path] [options]

    # lightest: small points, no validation (the defaults)
    .venv/Scripts/python.exe experiment/v1/_run_v1_viewer.py

Options:
    case                 case yaml path (default: cavity 1M)
    --device N           physical device index (default: auto-pick first
                         present-capable discrete GPU)
    --point-size F       rendered sprite radius (default 12.0; lower = cheaper)
    --validation         enable Vulkan validation layer (slower)
    --auto-quit SECONDS  auto-close after N seconds (smoke test)
    --window-width/-height  window size (default 1280x720)
    --log-fps PATH       append per-window fps samples (CSV) for benchmarking

Hotkeys (SphRendererV1): SPACE pause, 0..5 color mode (0 speed, 1 accel,
    2 density dev, 3 voxel id, 4 kernel sum, 5 vorticity ω_z), P ortho<->perspective,
    F frame-fit, +/- and [ / ] step-rate, ; / ' shrink/grow point size,
    , / . tune the current mode's color scale, ESC quit. Mouse: left drag orbit,
    middle drag pan, scroll zoom. Default color = speed (saturates at 1 m/s = the
    lid speed); press 5 for the vorticity contour view (warm = CCW, cool = CW),
    then , / . to dial the scale.
"""

import argparse
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import glfw

from utils.sph.case import load_case
from utils.sph.vulkan_context import VulkanContext

from experiment.v1 import compile_shaders_v1
from experiment.v1.utils.renderer_v1 import SphRendererV1
from experiment.v1.utils.simulator_v1 import SphSimulatorV1


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight single-GPU Vulkan + GLFW SPH viewer (V1).")
    parser.add_argument("case", nargs="?", default=DEFAULT_CASE,
                        help=f"case yaml path (default: {DEFAULT_CASE})")
    parser.add_argument("--device", type=int, default=None,
                        help="physical device index (default: auto-pick)")
    parser.add_argument("--point-size", type=float, default=12.0,
                        help="rendered sprite radius (default 12.0; lower=cheaper "
                             "fill rate; adjust live with ; / ' )")
    parser.add_argument("--color-mode", type=int, default=0,
                        choices=(0, 1, 2, 3, 4, 5),
                        help="initial color mode (0 speed .. 5 vorticity); "
                             "switch live with keys 0-5")
    parser.add_argument("--vorticity-scale", type=float, default=None,
                        help="initial ω_z color scale for mode 5 (default 0.02 "
                             "= saturate at |ω_z|=50/s); tune live with , / .")
    parser.add_argument("--validation", action="store_true",
                        help="enable Vulkan validation layer (slower)")
    parser.add_argument("--auto-quit", type=float, default=None,
                        help="auto-close window after N seconds (smoke test)")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--log-fps", type=str, default=None, metavar="PATH",
                        help="append per-window fps samples (CSV) to this file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Compile V1 shaders: compute kernels -> experiment/v1/shaders/spv/ AND the
    # V1-local render .vert/.frag -> experiment/v1/shaders/spv/render/ (the
    # latter carries the vorticity color mode; SphRendererV1 loads from there).
    compile_shaders_v1.compile_v1_shaders()

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    case = load_case(args.case)
    n_active = sum(s.vertices.shape[0] for s in case.particle_sources)
    print(f"\n[v1-viewer] loaded {args.case}")
    print(f"[v1-viewer]   active particles: {n_active:,}")
    print(f"[v1-viewer]   validation={'ON' if args.validation else 'OFF'}  "
          f"point_size={args.point_size}")
    if args.log_fps:
        print(f"[v1-viewer]   logging fps to {args.log_fps}")

    required_extensions = list(glfw.get_required_instance_extensions())
    create_kwargs = dict(
        application_name="sph_v1_viewer",
        enable_validation=args.validation,
        extra_instance_extensions=required_extensions,
        extra_device_extensions=["VK_KHR_swapchain"],
    )
    if args.device is not None:
        create_kwargs["device_index"] = args.device

    with VulkanContext.create(**create_kwargs) as ctx:
        # Single GPU: pid layout [1, POOL_SIZE], voxel layout [1, NX*NY*NZ].
        sim = SphSimulatorV1(ctx, case)
        try:
            sim.bootstrap()
            with SphRendererV1(sim,
                               window_width=args.window_width,
                               window_height=args.window_height) as viewer:
                viewer.point_size = args.point_size
                viewer.color_mode = args.color_mode
                if args.vorticity_scale is not None:
                    viewer.vorticity_scale = args.vorticity_scale
                viewer.run(log_fps_path=args.log_fps,
                           auto_quit_seconds=args.auto_quit)
            status = sim.readback_global_status()
            print(f"[v1-viewer] final: step={sim.step_count} "
                  f"alive={status['alive_particle_count']:,} "
                  f"(expected {n_active:,})")
        finally:
            sim.destroy()


if __name__ == "__main__":
    main()
