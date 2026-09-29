"""
compile_shaders_v1.py — compile all V1 compute shaders.

Mirrors compile_shaders.py at repo root but operates on
experiment/v1/shaders/*.comp and emits experiment/v1/shaders/spv/.

Uses the SAME glslc flags as the V0 compile path (`--target-env=vulkan1.2`,
`-O`, includes pointed at the V1 shader dir for `common.glsl` /
`helpers.glsl`). Same target env as V0 keeps SPIR-V version at 1.5, which
the application's VkApplicationInfo.apiVersion=Vulkan1.2 expects — avoids
"SPIR-V 1.6 vs Vulkan 1.2" validation errors that we hit when V1 was
mistakenly compiled with --target-env=vulkan1.3.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v1/compile_shaders_v1.py
"""

import glob
import os
import subprocess
import sys


GLSLC = os.environ.get("VULKAN_SDK", "C:/VulkanSDK/1.4.341.1") + "/Bin/glslc.exe"

V1_SHADER_DIR = os.path.dirname(os.path.abspath(__file__)) + "/shaders"
V1_SPV_DIR = V1_SHADER_DIR + "/spv"


# Extra SPIR-V builds of one source with preprocessor macros (2026-09-27):
#   force.comp -> force_scalar.comp.spv with FORCE_WITH_SCALARS=1.
# The plain build compiles the scalar-transport code out entirely. Guarding it
# only with spec constants left 0.23 ms (5 %) in force.comp on the 3 mm tank
# even with SCALAR_VEC4_COUNT = 0, i.e. the driver does not remove all of it at
# specialization; with the macro the plain build times exactly like before the
# scalar work (log/2026-09-27_scalar-transport.md, section 7).
# 2026-09-30: builds with the thin plate code (-DWITH_THIN_PLATES=1, see
# shaders/thin_plates.glsl), for the two kernels that treat the neighbours
# behind a plate as wall dummies. The default builds contain none of that code.
SOURCE_VARIANTS = {
    "force.comp": [("force_scalar.comp", ["-DFORCE_WITH_SCALARS=1"]),
                   ("force_plates.comp", ["-DWITH_THIN_PLATES=1"]),
                   ("force_scalar_plates.comp", ["-DFORCE_WITH_SCALARS=1", "-DWITH_THIN_PLATES=1"])],
    "density.comp": [("density_plates.comp", ["-DWITH_THIN_PLATES=1"])],
    "correction.comp": [("correction_plates.comp", ["-DWITH_THIN_PLATES=1"])],
}


def _run_glslc(source: str, output: str, extra_arguments=()) -> None:
    command = [
        GLSLC,
        "--target-env=vulkan1.2",
        "-O",
        "-I", V1_SHADER_DIR,
        *extra_arguments,
        source,
        "-o", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR compiling {os.path.basename(source)}:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)


def compile_v1_shaders() -> None:
    if not os.path.isfile(GLSLC):
        sys.exit(f"glslc not found at {GLSLC}. Set VULKAN_SDK env var "
                 f"to your install root.")
    if not os.path.isdir(V1_SHADER_DIR):
        sys.exit(f"V1 shader dir not found: {V1_SHADER_DIR}")
    os.makedirs(V1_SPV_DIR, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(V1_SHADER_DIR, "*.comp")))
    n_compiled = 0
    n_skipped = 0
    for source in sources:
        name = os.path.basename(source)
        if name.startswith("_"):
            print(f"[v1] skip {name} (underscore-prefixed)")
            n_skipped += 1
            continue
        output = os.path.join(V1_SPV_DIR, f"{name}.spv")
        print(f"[v1] {name}")
        _run_glslc(source, output)
        n_compiled += 1
        for variant_name, extra_arguments in SOURCE_VARIANTS.get(name, []):
            print(f"[v1] {variant_name}  ({name} {' '.join(extra_arguments)})")
            _run_glslc(source, os.path.join(V1_SPV_DIR, f"{variant_name}.spv"), extra_arguments)
            n_compiled += 1

    # Render shaders (.vert/.frag) for SphRendererV1. Compiled with the SAME
    # include dir so #include "common.glsl" resolves to V1's buffer layout —
    # the render shader then reads the exact fields the compute kernels write
    # (e.g. acceleration.w = vorticity ω_z for color mode 5).
    render_dir = os.path.join(V1_SHADER_DIR, "render")
    if os.path.isdir(render_dir):
        render_spv_dir = os.path.join(V1_SPV_DIR, "render")
        os.makedirs(render_spv_dir, exist_ok=True)
        render_sources = sorted(
            glob.glob(os.path.join(render_dir, "*.vert"))
            + glob.glob(os.path.join(render_dir, "*.frag")))
        for source in render_sources:
            name = os.path.basename(source)
            output = os.path.join(render_spv_dir, f"{name}.spv")
            print(f"[v1] render/{name}")
            _run_glslc(source, output)
            n_compiled += 1

    print(f"[v1] compiled {n_compiled} shaders, skipped {n_skipped}")


if __name__ == "__main__":
    compile_v1_shaders()
