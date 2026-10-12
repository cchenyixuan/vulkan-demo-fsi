"""_digitize_haringa2023_fig1.py - experimental points of Haringa (2023) Fig. 1
read from the PDF's vector graphics (2026-10-13).

Haringa, C. (2023) Eng. Life Sci. 23:e2100159, Figure 1 (page 6): Rushton
impeller outflow profiles at y = T/3 between the baffles, Jahoda geometry
(T = 0.29 m, D = C = T/3), 300 rpm.  The figure is vector art, so every marker
is a small path whose centre is the plotted value; no pixel picking is needed.
The three panels of the top row are read (the middle row carries the same
experimental markers at the same positions, checked here):

  panel 1  U_rad/U_tip     vs xi = (r - r_tip)/(R - r_tip)
  panel 2  k_t/U_tip^2     vs xi
  panel 3  eps/(N^3 D^2)   vs xi

Series (marker in the figure, source as cited by Haringa 2023):
  haringa2018   black squares, Haringa, Vandewijer, Mudde (2018) CERD 136:870,
                LDA, top impeller of a two-Rushton T = 0.26 m tank, N = 5.78 1/s;
                k_t with the blade-passage periodicity removed (their Eq. 5);
                panel 2 squares carry error bars (read as err_lo / err_hi)
  wu1989        yellow diamonds, Wu & Patterson (1989) CES 44:2207, LDA,
                k_t with the periodic component removed; eps from the energy
                balance / spectra
  murthy2008    open circles, Murthy & Joshi (2008) CES 63:5468, LDA
  ducci2005_lda open stars, Ducci & Yianneskis (2005) AIChE J 51:2133, LDA eps
  ducci2005_piv filled stars, same source, PIV eps
  ranade1990    grey crosses, velocity summary of Ranade & Joshi (1990)
                Trans IChemE 68:19 (three curves of the summary band, series 1-3)

Also read: the solid '360-BASE' LB-LES curve of the middle row (the thin filled
polygon of the line is reduced to its centre line, binned in xi).

Precision: marker centres are exact in the PDF coordinate frame; the axis
calibration uses the frame lines and ticks (0.36 pt wide on a 88 pt axis), so
the reading error is below 0.005 of the axis range in both coordinates.  The
remaining uncertainty is the original authors' plotting of their data.

Usage (repo root):
  python experiment/v1/checks/_digitize_haringa2023_fig1.py [--pdf PATH] [--out docs/data]
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    sys.exit("PyMuPDF (fitz) is needed: D:/Coding/Anaconda3/python.exe has it")

DEFAULT_PDF = (pathlib.Path(__file__).resolve().parents[4] / "reference" /
               "Haringa - 2023 - An analysis of organism lifelines in an industrial bioreactor "
               "using Lattice\u2010Boltzmann CFD.pdf")
PAGE_INDEX = 5
QUANTITIES = ("U_rad/U_tip", "k_t/U_tip^2", "eps/N3D2")
AXIS_TOP = (1.0, 0.15, 15.0)          # value at the top frame line of each panel
BLACK = (0.134, 0.121, 0.124)
YELLOW = (0.999, 0.780, 0.087)
GREY_CROSS = (0.692, 0.692, 0.688)
FRAME_GREY = (0.303, 0.304, 0.312)
# legend entries of panel 1 (top row): marker centre x and y in pt, excluded from the data
LEGEND_X = 197.6
LEGEND_Y = (57.4, 64.8, 73.4, 80.7, 89.3, 97.6)


def same(colour, reference, tol=0.02):
    return colour is not None and all(abs(a - b) < tol for a, b in zip(colour, reference))


def subpaths(drawing):
    """Split a drawing into connected runs of items (one marker each)."""
    paths, current, previous_end = [], [], None
    for item in drawing["items"]:
        kind = item[0]
        if kind == "re":
            if current:
                paths.append(current)
                current = []
            paths.append([item])
            previous_end = None
            continue
        if kind == "l":
            start, end = item[1], item[2]
        elif kind == "c":
            start, end = item[1], item[4]
        else:
            continue
        if previous_end is None or abs(start.x - previous_end.x) > 0.05 or abs(start.y - previous_end.y) > 0.05:
            if current:
                paths.append(current)
            current = []
        current.append(item)
        previous_end = end
    if current:
        paths.append(current)
    return paths


def vertices(path):
    points = []
    for item in path:
        if item[0] == "re":
            rect = item[1]
            points += [(rect.x0, rect.y0), (rect.x1, rect.y1)]
        else:
            points += [(q.x, q.y) for q in item[1:] if hasattr(q, "x")]
    return np.array(points)


def centre(path):
    pts = vertices(path)
    return (float((pts[:, 0].min() + pts[:, 0].max()) / 2.0), float((pts[:, 1].min() + pts[:, 1].max()) / 2.0),
            float(pts[:, 0].max() - pts[:, 0].min()), float(pts[:, 1].max() - pts[:, 1].min()))


def is_legend(cx, cy):
    return abs(cx - LEGEND_X) < 1.0 and any(abs(cy - y) < 1.0 for y in LEGEND_Y)


# ---------------------------------------------------------------------------
# Axis frames
# ---------------------------------------------------------------------------

def panel_frames(drawings, y_min, y_max):
    """Axis boxes of the three panels of one row: [(x0, x1, y_top, y_bottom)]
    from the frame-grey rectangles (axis lines are the long thin ones)."""
    verticals, horizontals = [], []
    for drawing in drawings:
        if not same(drawing.get("fill"), FRAME_GREY):
            continue
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            rect = item[1]
            if rect.y0 < y_min or rect.y1 > y_max:
                continue
            if rect.height > 50 and rect.width < 1:
                verticals.append(((rect.x0 + rect.x1) / 2, rect.y0, rect.y1))
            elif rect.width > 50 and rect.height < 1:
                horizontals.append(((rect.y0 + rect.y1) / 2, rect.x0, rect.x1))
    xs = []  # cluster the vertical lines (the frame and the tick strip overlap within 0.5 pt)
    for x in sorted(v[0] for v in verticals):
        if xs and abs(x - xs[-1]) < 0.5:
            continue
        xs.append(x)
    frames = []
    # pair the vertical lines: left and right of each panel are ~88 pt apart
    used = set()
    for left in xs:
        if left in used:
            continue
        right = [x for x in xs if 80 < x - left < 100]
        if not right:
            continue
        right = right[0]
        used.update((left, right))
        tops = [h[0] for h in horizontals if h[1] <= left + 1 and h[2] >= right - 1]
        frames.append((left, right, min(tops), max(tops)))
    frames.sort()
    if len(frames) != 3:
        raise RuntimeError(f"expected 3 panels, found {frames}")
    return frames


def to_data(frame, panel, cx, cy):
    x0, x1, y_top, y_bottom = frame
    xi = (cx - x0) / (x1 - x0)
    value = AXIS_TOP[panel] * (y_bottom - cy) / (y_bottom - y_top)
    return xi, value


def which_panel(frames, cx, cy, margin=3.0):
    for index, (x0, x1, y_top, y_bottom) in enumerate(frames):
        if x0 - margin <= cx <= x1 + margin and y_top - margin <= cy <= y_bottom + margin:
            return index
    return None


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def read_markers(drawings, frames, y_min, y_max):
    rows = []  # (source, panel, cx, cy, series)
    error_caps = []  # (cx, cy) of the error-bar caps of the squares (panel 2)
    for drawing in drawings:
        rect = drawing["rect"]
        if rect.y1 < y_min or rect.y0 > y_max:
            continue
        fill = drawing.get("fill")
        items = drawing["items"]
        kinds = "".join(item[0] for item in items)
        if same(fill, YELLOW) and kinds == "llll":
            cx, cy, w, h = centre(items)
            if 4.0 < w < 5.2 and not is_legend(cx, cy):
                rows.append(("wu1989", which_panel(frames, cx, cy), cx, cy, 0))
        elif same(fill, GREY_CROSS):
            seen, series, last_x = set(), 0, None
            for path in subpaths(drawing):
                if len(path) != 4:
                    continue
                cx, cy, w, h = centre(path)
                if is_legend(cx, cy) or any(abs(cx - x) < 0.5 and abs(cy - y) < 0.5 for x, y in seen):
                    continue  # each cross is two rotated rectangles; a few are drawn twice
                seen.add((cx, cy))
                if last_x is not None and cx > last_x + 8.0:
                    series += 1  # the summary's curves are drawn one after the other, right to left
                last_x = cx
                rows.append(("ranade1990", which_panel(frames, cx, cy), cx, cy, series))
        elif same(fill, BLACK):
            if kinds == "re":  # filled square (Haringa 2018), 3.6 x 3.5 pt
                cx, cy, w, h = centre(items)
                if 3.3 < w < 3.9 and 3.2 < h < 3.8 and not is_legend(cx, cy):
                    rows.append(("haringa2018", which_panel(frames, cx, cy), cx, cy, 0))
            elif kinds == "l" * 10:  # filled star (Ducci PIV)
                cx, cy, w, h = centre(items)
                if 4.0 < w < 4.8 and not is_legend(cx, cy):
                    rows.append(("ducci2005_piv", which_panel(frames, cx, cy), cx, cy, 0))
            else:
                paths = subpaths(drawing)
                if len(paths) >= 2 and all(len(p) == 27 for p in paths):  # open stars (Ducci LDA)
                    for path in paths:
                        cx, cy, w, h = centre(path)
                        rows.append(("ducci2005_lda", which_panel(frames, cx, cy), cx, cy, 0))
                elif len(paths) >= 2 and all(len(p) == 36 for p in paths):  # open circles (Murthy & Joshi)
                    for path in paths:
                        cx, cy, w, h = centre(path)
                        rows.append(("murthy2008", which_panel(frames, cx, cy), cx, cy, 0))
                elif kinds == "re" * len(items) and len(items) >= 12:  # error bars: caps + stems
                    for item in items:
                        r = item[1]
                        if r.width > 0.8 and r.height < 0.4:
                            error_caps.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    return rows, error_caps


def attach_error_bars(rows, error_caps, frames):
    """Pair the caps with the panel-2 squares by x; returns {row index: (lo, hi)} in data units."""
    bars = {}
    for index, (source, panel, cx, cy, series) in enumerate(rows):
        if source != "haringa2018" or panel != 1:
            continue
        caps = sorted(y for x, y in error_caps if abs(x - cx) < 0.6)
        if len(caps) == 2:
            hi = to_data(frames[1], 1, cx, caps[0])[1]
            lo = to_data(frames[1], 1, cx, caps[1])[1]
            bars[index] = (lo, hi)
    return bars


# ---------------------------------------------------------------------------
# LB-LES solid curve (360-BASE) of the middle row
# ---------------------------------------------------------------------------

def read_solid_curves(drawings, frames, y_min, y_max, bin_width=0.02):
    curves = {}
    for panel, frame in enumerate(frames):
        x0, x1, y_top, y_bottom = frame
        candidates = []
        for drawing in drawings:
            rect = drawing["rect"]
            if not same(drawing.get("fill"), BLACK):
                continue
            if rect.x0 < x0 - 2 or rect.x1 > x1 + 2 or rect.y0 < y_min or rect.y1 > y_max or rect.width < 60:
                continue
            paths = subpaths(drawing)
            if len(paths) == 1 and len(paths[0]) >= 100:
                candidates.append(paths[0])
        if not candidates:
            continue
        path = max(candidates, key=len)  # the solid line is one long run; dashed lines are many runs
        pts = vertices(path)
        xi = (pts[:, 0] - x0) / (x1 - x0)
        value = AXIS_TOP[panel] * (y_bottom - pts[:, 1]) / (y_bottom - y_top)
        edges = np.arange(0.0, 1.0 + bin_width, bin_width)
        centres, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (xi >= lo) & (xi < hi)
            if mask.sum() >= 2:
                centres.append(float(xi[mask].mean()))
                means.append(float(value[mask].mean()))
        curves[QUANTITIES[panel]] = (centres, means)
    return curves


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=pathlib.Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/data"))
    args = parser.parse_args()

    page = fitz.open(str(args.pdf))[PAGE_INDEX]
    drawings = page.get_drawings()
    top = panel_frames(drawings, 45, 150)
    middle = panel_frames(drawings, 175, 280)
    print("top-row frames (x0, x1, y_top, y_bottom):")
    for frame in top:
        print("   ", tuple(round(v, 2) for v in frame))

    rows, caps = read_markers(drawings, top, 45, 150)
    bars = attach_error_bars(rows, caps, top)
    rows_middle, _ = read_markers(drawings, middle, 175, 280)

    # consistency: the middle row repeats the same markers
    def as_data(rows_, frames_):
        out = {}
        for index, (source, panel, cx, cy, series) in enumerate(rows_):
            if panel is None:
                continue
            xi, value = to_data(frames_[panel], panel, cx, cy)
            out.setdefault((source, panel), []).append((xi, value / AXIS_TOP[panel], series, index))
        return out
    top_data, middle_data = as_data(rows, top), as_data(rows_middle, middle)
    worst = 0.0
    for key, values in top_data.items():
        other = middle_data.get(key, [])
        if len(other) != len(values):
            print(f"  note: {key} has {len(values)} markers in the top row and {len(other)} in the middle row")
            continue
        a = np.array(sorted((v[0], v[1]) for v in values))
        b = np.array(sorted((v[0], v[1]) for v in other))
        worst = max(worst, float(np.abs(a - b).max()))
    print(f"top vs middle row marker positions: largest difference {worst:.4f} of the axis range")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "haringa2023_fig1_experiments.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# Haringa (2023) Eng Life Sci 23:e2100159, Fig. 1 top row, experimental markers read from the PDF vector paths "
                         "by experiment/v1/checks/_digitize_haringa2023_fig1.py; reading error < 0.005 of the axis range"])
        writer.writerow(["source", "quantity", "xi", "value", "err_lo", "err_hi", "series"])
        count = {}
        for key in sorted(top_data, key=lambda k: (k[1], k[0])):
            source, panel = key
            for xi, value, series, index in sorted(top_data[key]):
                lo, hi = bars.get(index, ("", ""))
                writer.writerow([source, QUANTITIES[panel], f"{xi:.4f}", f"{value * AXIS_TOP[panel]:.4g}",
                                 f"{lo:.4g}" if lo != "" else "", f"{hi:.4g}" if hi != "" else "", series + 1])
                count[(source, QUANTITIES[panel])] = count.get((source, QUANTITIES[panel]), 0) + 1
    print(f"wrote {out_path}")
    for key, n in sorted(count.items()):
        print(f"   {key[0]:14s} {key[1]:12s} {n:3d} points")

    curves = read_solid_curves(drawings, middle, 175, 280)
    curve_path = args.out / "haringa2023_fig1_lbles_360base.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# Haringa (2023) Fig. 1 middle row, solid line 360-BASE (LB-LES, NX = 360), centre line of the "
                         "drawn stroke binned in xi (0.02); read by _digitize_haringa2023_fig1.py"])
        writer.writerow(["quantity", "xi", "value"])
        for quantity, (xs, ys) in curves.items():
            for xi, value in zip(xs, ys):
                writer.writerow([quantity, f"{xi:.4f}", f"{value:.4g}"])
    print(f"wrote {curve_path} ({', '.join(f'{q}: {len(v[0])} bins' for q, v in curves.items())})")


if __name__ == "__main__":
    main()
