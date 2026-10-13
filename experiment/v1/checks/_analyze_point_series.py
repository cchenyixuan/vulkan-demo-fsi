"""
_analyze_point_series.py — macro-instability analysis of the velocity time series at the mid-plane LDA
points of the 2-impeller tank (2026-10-14; Haringa, Vandewijer and Mudde 2018, Chem. Eng. Res. Des.
136:870, section 3.2.1, Tables 5 and 6), from the <stem>_series.npz of
experiment/v1/utils/flow_statistics.py (generator --point-series).

Per point (7 radial x 11 axial positions, the four baffle planes averaged unless --per-plane) and per
velocity component (radial, axial; the tangential component, which the LDA did not measure, is
evaluated too):

    mean U, variance u'^2 of the kernel-averaged signal u(t) sampled every `interval` s;
    auto-correlation  rho(tau) = <u'(t) u'(t + tau)> / u'^2  (biased estimator, FFT, lags up to
                      --max-lag s, averaged over the planes);
    fit of van Maanen (1999), the paper's Eq. (3), to rho(tau > 0):
        rho_fit(tau) = b + c0 exp(-a0 tau) + sum_{i=1,2} sum_{j=1,2} c_ij exp(-a_ij tau) cos(2 pi j f_i tau)
    two base frequencies f_1, f_2 (bounds --f-bounds, default 0.005 N .. 0.12 N) and their first
    harmonics, damping a_ij in [0, --a-max] (the paper's constraint 0.33 1/s), amplitudes c >= 0,
    started from a grid of (f_1, f_2) guesses, the lowest residual kept;
    noise fraction        epsilon = 1 - rho_fit(0) = 1 - (b + c0 + sum c_ij)
                          (the lag-0 value of a signal with white noise exceeds the fit);
    periodic fraction     P_fit = sum c_ij                          (Eq. (5))
    spectral alternative  P_SI = E(f < f_cut) / E_total of the Welch spectrum of u', f_cut = --f-cut
                          (default 0.1 N as the paper)                                  (Eq. (6))
    main frequency        f/N of the fitted cosine with the largest c_ij, and the spectral peak below f_cut.

Energies (two-component isotropy estimate of the paper, u'^2 the raw variance):
    k*   = 3/4 (u'^2_rad + u'^2_ax) (1 - epsilon)            total fluctuating energy without noise
    k_t  = 3/4 sum_c u'^2_c (1 - epsilon_c - P_c)              turbulent energy (MI removed)
    k_MI = k* - k_t                                            macro-instability energy
with P = P_fit (k_t,fit, k_MI,fit; Table 5) and P = P_SI (k_t,SI, k_MI,SI; Table 6); k3* and k3_t use all
three components with 1/2 sum. All energies are reported as 1000 k / U_tip^2.

Outputs (--out-dir, default next to the file): <label>_series_table.csv (one row per point and
component), <label>_series_summary.json, figures <label>_series_acf.png (rho and fit at y/T = 0 for
three radii, as Figure 4), <label>_series_spectra.png (u_ax spectra at y/T = 0, as Figure 5),
<label>_series_energy.png (k*, k_t, k_MI against y/T per radius, as Figure 3C).

usage (repo root):
    python experiment/v1/checks/_analyze_point_series.py STATS_series.npz --tank CASE/tank.json
        [--max-lag 20] [--f-cut-over-N 0.1] [--a-max 0.33] [--per-plane] [--out-dir DIR] [--label NAME]
    python experiment/v1/checks/_analyze_point_series.py --selftest      (synthetic signal, known components)
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

COMPONENTS = ("radial", "tangential", "axial")
FIT_COMPONENTS = ("radial", "axial")


def autocorrelation(signal, max_lag):
    """Biased auto-correlation coefficient rho(k), k = 0 .. max_lag, of a (samples,) or (samples, n)
    array along axis 0 (mean removed), by FFT."""
    fluctuation = signal - signal.mean(axis=0, keepdims=True)
    samples = fluctuation.shape[0]
    size = 1
    while size < 2 * samples:
        size *= 2
    spectrum = np.fft.rfft(fluctuation, n=size, axis=0)
    covariance = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=0)[:max_lag + 1] / samples
    variance = covariance[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return covariance / variance


def welch_spectrum(signal, interval, segment):
    """One-sided Welch power spectral density (Hann window, 50 % overlap) of the fluctuation of a
    (samples,) signal; returns frequencies and the density normalised so that trapz(S, f) = variance."""
    fluctuation = signal - signal.mean()
    samples = fluctuation.size
    segment = min(segment, samples)
    step = max(1, segment // 2)
    window = np.hanning(segment)
    accumulated = np.zeros(segment // 2 + 1)
    count = 0
    for start in range(0, samples - segment + 1, step):
        piece = fluctuation[start:start + segment] * window
        accumulated += np.abs(np.fft.rfft(piece)) ** 2
        count += 1
    frequency = np.fft.rfftfreq(segment, interval)
    density = accumulated / count * 2.0 * interval / np.sum(window ** 2)
    density[0] *= 0.5
    if segment % 2 == 0:
        density[-1] *= 0.5
    return frequency, density


def damped_cosines(parameters, tau, frequencies_count=2):
    """rho_fit(tau) of Eq. (3): parameters = [b, c0, a0, f_1, f_2, (c_11, a_11, c_12, a_12), (c_21, ...)]."""
    offset, amplitude0, damping0 = parameters[:3]
    value = offset + amplitude0 * np.exp(-damping0 * tau)
    base = 3 + frequencies_count
    for index in range(frequencies_count):
        frequency = parameters[3 + index]
        for harmonic in range(2):
            amplitude, damping = parameters[base + 4 * index + 2 * harmonic: base + 4 * index + 2 * harmonic + 2]
            value = value + amplitude * np.exp(-damping * tau) * np.cos(2.0 * math.pi * (harmonic + 1) * frequency * tau)
    return value


def fit_damped_cosines(tau, rho, speed, f_bounds, a_max, guesses):
    """Least-squares fit of Eq. (3) to rho(tau) for tau > 0, several starts; returns (parameters, cost)."""
    from scipy.optimize import least_squares
    keep = tau > 0.0
    tau, rho = tau[keep], rho[keep]
    lower = np.array([-0.3, 0.0, 0.0, f_bounds[0] * speed, f_bounds[0] * speed] + [0.0, 0.0] * 4)
    upper = np.array([0.5, 1.5, np.inf, f_bounds[1] * speed, f_bounds[1] * speed] + [1.0, a_max] * 4)
    # integral time scale of the turbulent part from the first crossing of rho = 1/e
    below = np.nonzero(rho < math.exp(-1.0))[0]
    damping0 = 1.0 / tau[below[0]] if below.size else 1.0 / tau[-1]
    best = None
    for guess_1, guess_2 in guesses:
        start = np.array([0.0, max(rho[0], 0.1), damping0, guess_1 * speed, guess_2 * speed] + [0.02, 0.05] * 4)
        start = np.minimum(np.maximum(start, lower + 1e-9), np.where(np.isfinite(upper), upper - 1e-9, start))
        try:
            solution = least_squares(lambda p: damped_cosines(p, tau) - rho, start, bounds=(lower, upper),
                                     max_nfev=4000)
        except ValueError:
            continue
        if best is None or solution.cost < best.cost:
            best = solution
    return best.x, best.cost


def fit_summary(parameters, speed):
    """epsilon, P_fit, the cosines (f/N, harmonic, c, a) sorted by amplitude, the main f/N."""
    offset, amplitude0 = parameters[0], parameters[1]
    cosines = []
    for index in range(2):
        frequency = parameters[3 + index]
        for harmonic in range(2):
            amplitude, damping = parameters[5 + 4 * index + 2 * harmonic: 5 + 4 * index + 2 * harmonic + 2]
            cosines.append(dict(f_over_N=float((harmonic + 1) * frequency / speed), base_f_over_N=float(frequency / speed),
                                harmonic=harmonic + 1, c=float(amplitude), a=float(damping)))
    periodic = float(sum(c["c"] for c in cosines))
    noise = float(1.0 - (offset + amplitude0 + periodic))
    # the base frequency and its harmonic are interchangeable for the fit (f_1 = 0.01 N with all the amplitude in
    # the harmonic at 0.02 N fits as well as f_1 = 0.02 N): report the cosines by amplitude, the main one and the
    # strongest one at a clearly different frequency (> 20 % apart)
    ranked = sorted(cosines, key=lambda c: c["c"], reverse=True)
    main = ranked[0]
    second = next((c for c in ranked[1:] if abs(c["f_over_N"] - main["f_over_N"]) > 0.2 * main["f_over_N"]), None)
    return dict(epsilon=noise, periodic=periodic, offset=float(offset), c0=float(amplitude0), a0=float(parameters[2]),
                cosines=cosines, main_f_over_N=main["f_over_N"], main_c=main["c"],
                second_f_over_N=second["f_over_N"] if second else float("nan"), second_c=second["c"] if second else 0.0,
                f1_over_N=float(parameters[3] / speed), f2_over_N=float(parameters[4] / speed))


def analyse_series(time, velocity, meta_entry, speed, tip_speed, args, label, out_dir):
    """velocity (samples, points, 3) in cylindrical components; meta_entry the generator's metadata."""
    interval = float(np.median(np.diff(time)))
    plane_index = np.asarray(meta_entry["plane_index"])
    r_index, y_index = np.asarray(meta_entry["r_index"]), np.asarray(meta_entry["y_index"])
    r_over_R, y_over_T = np.asarray(meta_entry["r_over_R"]), np.asarray(meta_entry["y_over_T"])
    planes = np.unique(plane_index)
    max_lag = int(round(args.max_lag / interval))
    samples = velocity.shape[0]
    if max_lag > samples // 2:
        max_lag = samples // 2
        print(f"  note: record of {samples} samples, max lag reduced to {max_lag * interval:.2f} s")
    tau = np.arange(max_lag + 1) * interval
    f_cut = args.f_cut_over_N * speed
    segment = int(round(min(args.max_lag * 2.0, samples * interval / 2.0) / interval))
    # starts of the fit: a grid of (f_1, f_2) guesses around the paper's frequencies, plus the two highest
    # spectral peaks below f_cut of the point itself (added per point)
    guesses = [(g1, g2) for g1 in (0.012, 0.015, 0.02, 0.025, 0.03) for g2 in (0.04, 0.05, 0.055, 0.065, 0.08)]
    # the fit needs at least one period of the slow MI (f/N = 0.02, 10 s at N = 5) within the lag range
    can_fit = max_lag * interval >= 0.999 / (0.02 * speed) and samples >= 50
    if not can_fit:
        print(f"  note: record too short for the periodic fit ({samples} samples, {max_lag * interval:.2f} s of lag): "
              f"means and rms only")

    rows = []
    groups = [(None, planes)] if not args.per_plane else [(int(p), np.array([p])) for p in planes]
    acf_store = {}
    spectra_store = {}
    for group_label, group_planes in groups:
        for radial in range(r_over_R.size):
            for axial in range(y_over_T.size):
                chosen = np.nonzero((r_index == radial) & (y_index == axial) & np.isin(plane_index, group_planes))[0]
                if chosen.size == 0:
                    continue
                signals = velocity[:, chosen, :]                      # (samples, planes, 3)
                if np.isnan(signals).any():
                    good = ~np.isnan(signals).any(axis=(0, 2))
                    signals = signals[:, good, :]
                    if signals.shape[1] == 0:
                        continue
                row = dict(group=group_label, r_over_R=float(r_over_R[radial]), y_over_T=float(y_over_T[axial]),
                           planes=int(signals.shape[1]))
                variances = {}
                for component_index, component in enumerate(COMPONENTS):
                    signal = signals[:, :, component_index]
                    mean = float(signal.mean())
                    variance = float(((signal - signal.mean(axis=0, keepdims=True)) ** 2).mean())
                    row[f"mean_{component}"] = mean
                    row[f"rms_{component}"] = math.sqrt(variance)
                    variances[component] = variance
                    if component not in FIT_COMPONENTS:
                        continue
                    rho = np.nanmean(autocorrelation(signal, max_lag), axis=1)
                    # spectral method: Welch spectrum of every plane, averaged
                    densities = []
                    for plane in range(signal.shape[1]):
                        frequency, density = welch_spectrum(signal[:, plane], interval, segment)
                        densities.append(density)
                    density = np.mean(densities, axis=0)
                    total = np.trapz(density[1:], frequency[1:])
                    low = frequency <= f_cut
                    low_energy = np.trapz(density[low][1:], frequency[low][1:]) if low.sum() > 2 else 0.0
                    row[f"P_SI_{component}"] = float(low_energy / total) if total > 0 else float("nan")
                    peak = frequency[1:][np.argmax(density[1:][low[1:]])] if low[1:].any() else float("nan")
                    row[f"f_peak_over_N_{component}"] = float(peak / speed)
                    if can_fit:
                        order = np.argsort(density[1:][low[1:]])[::-1]
                        peaks = sorted(frequency[1:][low[1:]][order[:2]] / speed)
                        own = guesses + ([(peaks[0], peaks[1])] if len(peaks) == 2 and peaks[0] < peaks[1] else [])
                        parameters, cost = fit_damped_cosines(tau, rho, speed, args.f_bounds, args.a_max, own)
                        summary = fit_summary(parameters, speed)
                        row[f"epsilon_{component}"] = summary["epsilon"]
                        row[f"P_fit_{component}"] = summary["periodic"]
                        row[f"f1_over_N_{component}"] = summary["f1_over_N"]
                        row[f"f2_over_N_{component}"] = summary["f2_over_N"]
                        row[f"f_main_over_N_{component}"] = summary["main_f_over_N"]
                        row[f"f_second_over_N_{component}"] = summary["second_f_over_N"]
                        row[f"c_main_{component}"] = summary["main_c"]
                        row[f"c_second_{component}"] = summary["second_c"]
                        row[f"cosines_{component}"] = summary["cosines"]
                        row[f"fit_cost_{component}"] = float(cost)
                        if group_label is None and axial == int(np.argmin(np.abs(y_over_T))):
                            acf_store[(radial, component)] = (tau, rho, damped_cosines(parameters, tau))
                    else:
                        for key in ("epsilon", "P_fit", "f1_over_N", "f2_over_N", "f_main_over_N", "f_second_over_N",
                                    "c_main", "c_second"):
                            row[f"{key}_{component}"] = float("nan")
                    if group_label is None and axial == int(np.argmin(np.abs(y_over_T))) and component == "axial":
                        spectra_store[radial] = (frequency, density)
                # energies (1000 k / U_tip^2)
                scale = 1e3 / tip_speed ** 2
                noise = {c: row.get(f"epsilon_{c}", float("nan")) for c in FIT_COMPONENTS}
                row["k_star"] = scale * 0.75 * sum(variances[c] * (1.0 - noise[c]) for c in FIT_COMPONENTS)
                row["k_raw2"] = scale * 0.75 * sum(variances[c] for c in FIT_COMPONENTS)
                row["k_raw3"] = scale * 0.5 * sum(variances.values())
                for method in ("fit", "SI"):
                    periodic = {c: row.get(f"P_{method}_{c}", float("nan")) for c in FIT_COMPONENTS}
                    row[f"k_t_{method}"] = scale * 0.75 * sum(variances[c] * (1.0 - noise[c] - periodic[c])
                                                               for c in FIT_COMPONENTS)
                    row[f"k_MI_{method}"] = row["k_star"] - row[f"k_t_{method}"]
                rows.append(row)

    # --- tables ---------------------------------------------------------------------------------------
    centre = rows_at(rows, None, 0.0) if not args.per_plane else []
    if centre:
        print(f"  plane y/T = 0 (mean of the baffle planes), {samples} samples every {interval:.4f} s, "
              f"lag {max_lag * interval:.1f} s, f_cut/N = {args.f_cut_over_N}; velocities in m/s, U_tip {tip_speed:.4f} m/s")
        print("  Table 5 form (periodic fit):")
        print("   r/R     u_rad'  eps_rad  P_rad  fmain/N (c)   f2nd/N (c)  |  u_ax'  eps_ax   P_ax  fmain/N (c)   f2nd/N (c)  | "
              "1000k*  1000kt  1000kMI")
        for row in centre:
            print(f"   {row['r_over_R']:5.3f}  {row['rms_radial']:6.3f}  {row['epsilon_radial']:6.3f} {row['P_fit_radial']:6.3f} "
                  f"{row['f_main_over_N_radial']:6.3f} ({row['c_main_radial']:5.3f}) {row['f_second_over_N_radial']:6.3f} "
                  f"({row['c_second_radial']:5.3f}) | "
                  f"{row['rms_axial']:6.3f}  {row['epsilon_axial']:6.3f} {row['P_fit_axial']:6.3f} "
                  f"{row['f_main_over_N_axial']:6.3f} ({row['c_main_axial']:5.3f}) {row['f_second_over_N_axial']:6.3f} "
                  f"({row['c_second_axial']:5.3f}) | "
                  f"{row['k_star']:6.2f}  {row['k_t_fit']:6.2f}  {row['k_MI_fit']:7.2f}")
        print("  Table 6 form (spectral integration, f/N < f_cut/N):")
        print("   r/R    (Ep/Et)rad  (Ep/Et)ax  fpeak/N,rad  fpeak/N,ax  1000kt,SI  1000kMI,SI  | 1000k_raw2  1000k_raw3")
        for row in centre:
            print(f"   {row['r_over_R']:5.3f}   {row['P_SI_radial']:8.3f}   {row['P_SI_axial']:8.3f}     "
                  f"{row['f_peak_over_N_radial']:6.3f}      {row['f_peak_over_N_axial']:6.3f}    {row['k_t_SI']:7.2f}   "
                  f"{row['k_MI_SI']:8.2f}  |   {row['k_raw2']:7.2f}    {row['k_raw3']:7.2f}")
        print("  reference (LDA, Table 5, N = 5.78 1/s, U_tip 1.574 m/s): u_rad' 0.149 .. 0.083 m/s, u_ax' 0.192 .. 0.055,")
        print("    f1/N 0.015-0.020, f2/N 0.052-0.061, 1000 k_t,fit/U_tip^2 6.2 .. 10.8, 1000 k_MI,fit 0.4 .. 2.4;")
        print("    Table 6: (Ep/Et)ax 0.14 .. 0.55, 1000 k_t,SI 4.7 .. 6.8, 1000 k_MI,SI 2.2 .. 6.5")
    for key, title in (("k_star", "1000 k* / U_tip^2 (noise removed)"), ("k_t_fit", "1000 k_t,fit / U_tip^2"),
                       ("k_MI_fit", "1000 k_MI,fit / U_tip^2"), ("k_MI_SI", "1000 k_MI,SI / U_tip^2"),
                       ("f_main_over_N_axial", "main f/N of u_ax (fit)")):
        grid = np.full((r_over_R.size, y_over_T.size), np.nan)
        for row in rows:
            if row["group"] is None or args.per_plane and row["group"] == 0:
                grid[np.argmin(np.abs(r_over_R - row["r_over_R"])), np.argmin(np.abs(y_over_T - row["y_over_T"]))] = row[key]
        print(f"  {title}: rows r/R, columns y/T")
        print("     r/R  " + "".join(f"{y:8.3f}" for y in y_over_T))
        for radial, r in enumerate(r_over_R):
            print(f"   {r:5.3f}  " + "".join(f"{grid[radial, column]:8.3f}" for column in range(y_over_T.size)))

    # --- files ------------------------------------------------------------------------------------------
    keys = ["group", "r_over_R", "y_over_T", "planes"] + [f"{k}_{c}" for c in COMPONENTS for k in ("mean", "rms")]
    keys += [f"{k}_{c}" for c in FIT_COMPONENTS for k in ("epsilon", "P_fit", "P_SI", "f1_over_N", "f2_over_N",
                                                           "f_main_over_N", "c_main", "f_second_over_N", "c_second",
                                                           "f_peak_over_N")]
    keys += ["k_raw2", "k_raw3", "k_star", "k_t_fit", "k_MI_fit", "k_t_SI", "k_MI_SI"]
    with open(out_dir / f"{label}_series_table.csv", "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join("" if row.get(k) is None else (f"{row[k]:.6e}" if isinstance(row[k], float) else str(row[k]))
                                  for k in keys) + "\n")
    summary = dict(label=label, samples=samples, interval=interval, max_lag=max_lag * interval, f_cut_over_N=args.f_cut_over_N,
                   a_max=args.a_max, f_bounds=list(args.f_bounds), rows=[{k: v for k, v in row.items()} for row in rows])
    (out_dir / f"{label}_series_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    plot(out_dir, label, rows, r_over_R, y_over_T, acf_store, spectra_store, speed, args)
    print(f"  wrote {out_dir / (label + '_series_table.csv')}, _series_summary.json and figures")
    return rows


def rows_at(rows, group, y_over_T):
    return sorted([row for row in rows if row["group"] == group and abs(row["y_over_T"] - y_over_T) < 1e-9],
                  key=lambda row: row["r_over_R"])


def plot(out_dir, label, rows, r_over_R, y_over_T, acf_store, spectra_store, speed, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if acf_store:
        chosen = [radial for radial in (0, 3, 5) if (radial, "axial") in acf_store]
        figure, axes = plt.subplots(2, len(chosen), figsize=(5 * len(chosen), 7), squeeze=False)
        for column, radial in enumerate(chosen):
            for line, component in enumerate(("axial", "radial")):
                tau, rho, fit = acf_store[(radial, component)]
                axis = axes[line, column]
                axis.plot(tau * speed, rho, lw=0.8, label="rho")
                axis.plot(tau * speed, fit, lw=1.2, label="fit, Eq. (3)")
                axis.set_xlabel("tau N (revolutions)")
                axis.set_title(f"u_{'ax' if component == 'axial' else 'rad'}, r/R = {r_over_R[radial]:.3f}, y/T = 0")
                axis.grid(alpha=0.3)
                axis.legend(fontsize=8)
        figure.suptitle(f"{label}: auto-correlation and damped-cosine fit")
        figure.tight_layout()
        figure.savefig(out_dir / f"{label}_series_acf.png", dpi=130)
        plt.close(figure)
    if spectra_store:
        figure, axis = plt.subplots(figsize=(7, 5))
        for radial, (frequency, density) in sorted(spectra_store.items()):
            keep = frequency > 0
            axis.loglog(frequency[keep] / speed, density[keep] * speed, lw=0.9, label=f"r/R = {r_over_R[radial]:.3f}")
        axis.axvline(args.f_cut_over_N, color="k", ls=":", lw=0.8)
        reference = np.array([0.01, 1.0])
        axis.loglog(reference, 1e-3 * reference ** (-5.0 / 3.0), "k--", lw=0.8, label="(f/N)^-5/3")
        axis.set_xlabel("f / N"); axis.set_ylabel("S(f) N / 1 [m^2/s^2]")
        axis.set_title(f"{label}: spectra of u_ax at y/T = 0")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(out_dir / f"{label}_series_spectra.png", dpi=130)
        plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, key, title in ((axes[0], "k_star", "1000 k* / U_tip^2"), (axes[1], "k_t_fit", "1000 k_t,fit / U_tip^2"),
                             (axes[2], "k_MI_fit", "1000 k_MI,fit / U_tip^2")):
        for radial, r in enumerate(r_over_R):
            points = sorted([(row["y_over_T"], row[key]) for row in rows
                             if row["group"] in (None, 0) and abs(row["r_over_R"] - r) < 1e-9])
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=f"r/R = {r:.3f}")
        axis.set_xlabel("y / T"); axis.set_title(title); axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    figure.suptitle(f"{label}: fluctuating energy in the mid-compartment plane")
    figure.tight_layout()
    figure.savefig(out_dir / f"{label}_series_energy.png", dpi=130)
    plt.close(figure)


def selftest(args):
    """Synthetic record: turbulence (AR(1), integral time 0.3 s) + two damped cosines at f/N = 0.02 and
    0.055 + white noise, at 2 x 2 points in 4 planes; the fit must recover the frequencies and the
    energy fractions."""
    rng = np.random.default_rng(1)
    speed, tip_speed, interval, samples = 5.0, 1.5184, 0.02, 6000
    time = np.arange(samples) * interval
    planes, radii, heights = 4, 2, 2
    count = planes * radii * heights
    sigma_t, amplitude_1, amplitude_2, sigma_noise = 0.10, 0.06, 0.05, 0.03
    velocity = np.zeros((samples, count, 3), dtype=np.float32)
    phi = math.exp(-interval / 0.3)
    for point in range(count):
        for component in range(3):
            turbulence = np.zeros(samples)
            noise = rng.standard_normal(samples)
            for index in range(1, samples):
                turbulence[index] = phi * turbulence[index - 1] + math.sqrt(1 - phi * phi) * noise[index]
            phase_1, phase_2 = rng.uniform(0, 2 * math.pi, 2)
            jitter_1 = np.cumsum(rng.standard_normal(samples)) * 0.02      # slow phase wander
            signal = (sigma_t * turbulence + amplitude_1 * np.cos(2 * math.pi * 0.02 * speed * time + phase_1 + jitter_1)
                      + amplitude_2 * np.cos(2 * math.pi * 0.055 * speed * time + phase_2)
                      + sigma_noise * rng.standard_normal(samples) + 0.2 * component)
            velocity[:, point, component] = signal
    expected_variance = sigma_t ** 2 + 0.5 * amplitude_1 ** 2 + 0.5 * amplitude_2 ** 2 + sigma_noise ** 2
    expected_periodic = (0.5 * amplitude_1 ** 2 + 0.5 * amplitude_2 ** 2) / expected_variance
    expected_noise = sigma_noise ** 2 / expected_variance
    meta_entry = dict(name="selftest", plane_index=[p for p in range(planes) for _ in range(radii * heights)],
                      r_index=[r for _ in range(planes) for r in range(radii) for _ in range(heights)],
                      y_index=[y for _ in range(planes) for _ in range(radii) for y in range(heights)],
                      r_over_R=[0.2, 0.5], y_over_T=[0.0, 0.1])
    out_dir = pathlib.Path(args.out_dir or "output/point_series_selftest")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"selftest: expected periodic fraction {expected_periodic:.3f}, noise fraction {expected_noise:.3f}, "
          f"f/N 0.020 and 0.055, rms {math.sqrt(expected_variance):.4f} m/s")
    rows = analyse_series(time, velocity, meta_entry, speed, tip_speed, args, "selftest", out_dir)
    # pass: every point finds the main frequency, the periodic and noise fractions within tolerance; the second
    # frequency (a local-minimum problem of the 13-parameter fit) in at least three quarters of the cases
    ok, second_hits, cases = True, 0, 0
    for row in rows:
        for component in FIT_COMPONENTS:
            main, second = row[f"f_main_over_N_{component}"], row[f"f_second_over_N_{component}"]
            frequencies = sorted((main, second))
            good = (min(abs(main - 0.02), abs(main - 0.055)) < 0.004
                    and abs(row[f"P_fit_{component}"] - expected_periodic) < 0.12
                    and abs(row[f"epsilon_{component}"] - expected_noise) < 0.06)
            second_good = abs(frequencies[0] - 0.02) < 0.004 and abs(frequencies[1] - 0.055) < 0.006
            print(f"  r/R {row['r_over_R']} y/T {row['y_over_T']} {component}: f/N {frequencies[0]:.4f} {frequencies[1]:.4f}, "
                  f"P_fit {row[f'P_fit_{component}']:.3f} (P_SI {row[f'P_SI_{component}']:.3f}), eps {row[f'epsilon_{component}']:.3f}"
                  f" -> {'ok' if good and second_good else ('main ok' if good else 'MISS')}")
            ok &= good
            second_hits += second_good
            cases += 1
    ok &= second_hits >= 0.75 * cases
    print(f"selftest {'PASSED' if ok else 'FAILED'} (second frequency found in {second_hits} of {cases})")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("series", nargs="?", help="<stem>_series.npz of the flow statistics")
    parser.add_argument("--tank", default=None, help="tank.json of the case (N, U_tip)")
    parser.add_argument("--max-lag", type=float, default=20.0, help="largest lag of the auto-correlation (s)")
    parser.add_argument("--f-cut-over-N", type=float, default=0.1, help="MI / turbulence cut-off of the spectral method")
    parser.add_argument("--f-bounds", type=float, nargs=2, default=(0.005, 0.12), metavar=("LOW", "HIGH"),
                        help="bounds of the fitted base frequencies, in f/N")
    parser.add_argument("--a-max", type=float, default=0.33, help="largest damping of a cosine (1/s), the paper's 0.33")
    parser.add_argument("--per-plane", action="store_true", help="fit every baffle plane separately")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest(args)
    if not args.series or not args.tank:
        parser.error("SERIES and --tank are required")
    data = np.load(args.series)
    meta = json.loads(str(data["meta"]))
    tank = json.loads(pathlib.Path(args.tank).read_text(encoding="utf-8"))
    speed, tip_speed = tank["rpm"] / 60.0, tank["tip_speed"]
    label = args.label or pathlib.Path(args.series).stem.replace("_series", "")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(args.series).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    time = np.asarray(data["time"])
    print(f"{label}: {meta['samples']} samples every {meta['interval']:.4f} s, t = {time[0]:.2f} .. {time[-1]:.2f} s, "
          f"kernel radius {meta['kernel_radius'] * 1e3:.1f} mm; N {speed:g} 1/s, U_tip {tip_speed:.4f} m/s")
    for entry in meta["series"]:
        velocity = np.asarray(data[entry["name"] + "/velocity"], dtype=np.float64)
        print(f"  series {entry['name']}: {velocity.shape[1]} points, {len(np.unique(entry['plane_index']))} planes")
        analyse_series(time, velocity, entry, speed, tip_speed, args,
                       label if len(meta["series"]) == 1 else f"{label}_{entry['name']}", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
