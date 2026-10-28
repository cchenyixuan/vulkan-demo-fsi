"""_vorticity_viewer_data.py — pack the vorticity grids of _vorticity_grid_30l.py for the 3D web viewer (2026-10-29).

Each grid becomes one grayscale PNG atlas: log10 |omega| quantised to 1..255 over [--log-low, --log-high]
(0 = masked), the z slices laid out in rows of --tiles-per-row tiles, each tile nx wide and ny high (row j of a
tile = grid index j along y, column i = index i along x). manifest.json lists the series, their frames and the
grid, so the page can rebuild the volume (index i + nx (j + ny k)).

usage (repo root):
    python experiment/v1/checks/_vorticity_viewer_data.py OUT_DIR \
        --series fluent "Fluent LES" native:output/vorticity30l/fluent_s3_t*.npz matched:output/vorticity30l/fluent_s6_t*.npz \
        --series sph "SPH 2 mm" native:... matched:...
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
from PIL import Image


def atlas(log_vorticity, low, high, tiles_per_row):
    nx, ny, nz = log_vorticity.shape
    value = log_vorticity.astype(np.float32)
    quantised = np.where(np.isfinite(value), np.clip(np.rint((value - low) / (high - low) * 254.0) + 1.0, 1, 255), 0)
    quantised = quantised.astype(np.uint8)
    rows = -(-nz // tiles_per_row)
    image = np.zeros((rows * ny, tiles_per_row * nx), np.uint8)
    for k in range(nz):
        row, column = divmod(k, tiles_per_row)
        image[row * ny:(row + 1) * ny, column * nx:(column + 1) * nx] = quantised[:, :, k].T
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--series", nargs="+", action="append", required=True, metavar="ITEM",
                        help="KEY LABEL VARIANT:GLOB [VARIANT:GLOB ...]")
    parser.add_argument("--log-low", type=float, default=-0.3, help="log10 |omega| at level 1 (1/s)")
    parser.add_argument("--log-high", type=float, default=2.7, help="log10 |omega| at level 255 (1/s)")
    parser.add_argument("--tiles-per-row", type=int, default=14)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"log_range": [arguments.log_low, arguments.log_high], "tiles_per_row": arguments.tiles_per_row,
                "series": []}
    for item in arguments.series:
        key, label, variants = item[0], item[1], item[2:]
        entry = {"key": key, "label": label, "variants": {}}
        for variant_spec in variants:
            variant, pattern = variant_spec.split(":", 1)
            frames = []
            for path in sorted(glob.glob(pattern), key=lambda p: float(np.load(p)["time"])):
                data = np.load(path)
                grid = data["log_vorticity"]
                if "grid" not in manifest:
                    edges = [data["x_edges"], data["y_edges"], data["z_edges"]]
                    manifest["grid"] = {"dims": list(grid.shape), "cell": float(edges[0][1] - edges[0][0]),
                                        "low": [float(edge[0]) for edge in edges], "high": [float(edge[-1]) for edge in edges]}
                elif list(grid.shape) != manifest["grid"]["dims"]:
                    raise SystemExit(f"{path}: grid {grid.shape} differs from {manifest['grid']['dims']}")
                time = float(data["time"])
                name = f"{key}_{variant}_t{time:05.1f}.png"
                Image.fromarray(atlas(grid, arguments.log_low, arguments.log_high, arguments.tiles_per_row), "L").save(
                    out / name, optimize=True)
                percentiles = [round(float(value), 2) for value in data["percentiles"]]
                frames.append({"time": time, "file": name, "percentiles_5_25_50_75_95_99_999": percentiles})
                print(f"{key} {variant} t {time:5.1f}: {name} {(out / name).stat().st_size / 1e6:.2f} MB, |omega| p50 "
                      f"{percentiles[2]} p95 {percentiles[4]} p99.9 {percentiles[6]} 1/s")
            entry["variants"][variant] = frames
        manifest["series"].append(entry)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
