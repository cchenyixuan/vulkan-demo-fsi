"""_check_plate_drag_2d.py — drag of an impulsively started flat plate normal to its motion, two-dimensional,
as a resolution series of the thin-plate (side-aware mirror) treatment (2026-10-13).

Motivation: the Rushton blade torque of the 30 L tank rises with every refinement (3 / 2 / 1.5 / 1 mm, the blade
19.2 mm high = 6 .. 19 dx) and the growth sits on the suction face (log/2026-10-13_blade-pressure-profiles.md).
A plate normal to a stream has a bounded drag coefficient, so a problem with a known answer tells at which
resolution (plate length / dx) the plate load saturates and whether the saturated value is right.

Problem (Koumoutsakos & Shiels 1996, J. Fluid Mech. 328:177; Xu & Nitsche 2014 J. Fluid Mech. 756:689):
a plate of length L = 2a and zero thickness in fluid at rest starts impulsively with the speed U normal to
itself, Re = U L / nu. The drag coefficient C_D = F / (1/2 rho U^2 L) (per unit depth, whole plate) decays from
the initial singularity (~ t^-1/2) to ~1.7 at t* = t U / L = 3 and ~1.25 at t* = 5 for Re 1000 (KS, read from
figure 20(b) of Xu & Nitsche, +- 0.1 on the half-plate value).

Realisation in this solver: there is no translating wall, only the rotor (rigid rotation about `rotor.pivot`).
The plate is the lattice line y = 0, |x| < a, and the pivot sits at (-R, 0) with R = `--radius-ratio` * a
(default 10): the plate moves along the circle of radius R with the speed U = Omega R at its centre, normal to
itself (a radial plate moves tangentially); the speed varies by +- a / R = +- 10 % along the plate. The box is
large enough for the arc the plate sweeps up to t* = `--t-end` plus `--margin` plate lengths on every side.

Numerical parameters follow the production configuration of the tank in dimensionless form: h = 3 dx,
c0 = 20 U, background pressure p_b = `--background-ratio` * 1/2 rho U^2 (the tank: 2000 Pa / 500 Pa = 4),
delta 0.1, PST 0.1, pair correction reverse, solid pressure accumulate, reaction forces, plate thickness
`--thickness-ratio` * L (the Rushton blade: 2.2 mm / 19.2 mm = 0.115).

Output per resolution: the history of the normal force on the plate (sum of the reaction records, divided by
the 2-D mass factor), its pressure-side and suction-side parts (sign of (x_i - c) . n), C_D(t*), and the mean
face pressures. Reference values of KS are printed next to the C_D at the requested times.

Usage (repo root, solver environment):
    python experiment/v1/checks/_check_plate_drag_2d.py --plate-cells 10 20 40 80
    python experiment/v1/checks/_check_plate_drag_2d.py --plate-cells 40 --hdx 5 --name-suffix _hdx5
"""
import argparse
import math
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sph.case import load_case, _calibrate_particle_volume      # noqa: E402
from utils.sph.vulkan_context import VulkanContext                    # noqa: E402
from experiment.v1 import compile_shaders_v1                          # noqa: E402
from experiment.v1.utils.simulator_v1 import SphSimulatorV1           # noqa: E402

REST_DENSITY = 1000.0
POWER = 7.0

# Koumoutsakos & Shiels 1996, Re 1000, whole-plate C_D = 2 F_D with F_D the half-plate value of
# Xu & Nitsche 2014 figure 20(b) (dashed curve "KS"), read off the figure, +- 0.2 on C_D.
REFERENCE_KS_RE1000 = {0.5: 3.5, 1.0: 3.1, 1.5: 2.6, 2.0: 2.3, 2.5: 2.0, 3.0: 1.76, 4.0: 1.44, 5.0: 1.24}
# Xu & Nitsche 2014, their finest run of the drag (h = 1/320, same figure), whole plate.
REFERENCE_XN_RE1000 = {0.5: 2.6, 1.0: 2.1, 1.5: 1.85, 2.0: 1.7, 2.5: 1.6, 3.0: 1.5}


def write_points(path, points):
    np.savetxt(path, points, fmt="v %.7f %.7f %.7f", header=f"# {points.shape[0]} particles", comments="")


def build_case(directory, arguments, plate_cells):
    """Two-dimensional box with the plate on the lattice line y = 0, |x| < a; rotor pivot at (-R, 0)."""
    directory.mkdir(parents=True, exist_ok=True)
    half_length = 0.5 * arguments.plate_length                    # a
    dx = arguments.plate_length / plate_cells
    hdx = arguments.hdx
    speed = arguments.speed                                        # U at the plate centre
    radius = arguments.radius_ratio * half_length                  # R
    omega = speed / radius
    t_end = arguments.t_end * arguments.plate_length / speed
    theta_end = omega * t_end
    # arc of the plate centre: (-R + R cos theta, R sin theta), theta in [0, theta_end]; plate half-length a
    margin = arguments.margin * arguments.plate_length
    x_low = -radius + radius * math.cos(theta_end) - half_length - margin
    x_high = half_length + margin
    y_low = -margin
    y_high = radius * math.sin(theta_end) + half_length + margin
    # lattice with a site line on y = 0 and a site column boundary symmetric about x = 0
    n_x_neg = int(math.ceil(-x_low / dx)); n_x_pos = int(math.ceil(x_high / dx))
    n_y_neg = int(math.ceil(-y_low / dx)); n_y_pos = int(math.ceil(y_high / dx))
    xs = (np.arange(-n_x_neg, n_x_pos) + 0.5) * dx
    ys = np.arange(-n_y_neg, n_y_pos + 1) * dx                    # includes y = 0 exactly
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    layers_wall = int(math.ceil(hdx))
    xs_ext = (np.arange(-n_x_neg - layers_wall, n_x_pos + layers_wall) + 0.5) * dx
    ys_ext = np.arange(-n_y_neg - layers_wall, n_y_pos + 1 + layers_wall) * dx
    shell = np.stack(np.meshgrid(xs_ext, ys_ext, indexing="ij"), axis=-1).reshape(-1, 2)
    inside = (shell[:, 0] > xs[0] - 0.5 * dx) & (shell[:, 0] < xs[-1] + 0.5 * dx) & \
             (shell[:, 1] > ys[0] - 0.5 * dx) & (shell[:, 1] < ys[-1] + 0.5 * dx)
    wall = shell[~inside]
    on_line = np.abs(grid[:, 1]) < 1e-6 * dx
    half_a_sites = 0.5 * plate_cells                               # plate = plate_cells sites along x
    plate_mask = on_line & (np.abs(grid[:, 0]) < half_a_sites * dx - 1e-6 * dx)
    plate = grid[plate_mask]
    fluid = grid[~plate_mask]

    def lift(points):
        return np.column_stack([points, np.zeros(points.shape[0])])
    write_points(directory / "fluid.obj", lift(fluid))
    write_points(directory / "wall.obj", lift(wall))
    write_points(directory / "plate.obj", lift(plate))
    lo = shell.min(axis=0) - 0.6 * dx
    hi = shell.max(axis=0) + 0.6 * dx
    with open(directory / "frame.obj", "w") as handle:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (-0.5 * dx, 0.5 * dx):
                    handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
    point_count = fluid.shape[0] + wall.shape[0] + plate.shape[0]
    pool = int(math.ceil(point_count * 1.15 / 128) * 128)
    dynamic_pressure = 0.5 * REST_DENSITY * speed ** 2
    case = {
        "schema_version": 2,
        "time": {"total": None, "max_steps": None, "output_cadence": None},
        "physics": {"dimension": 2, "h": hdx * dx, "particle_radius": 0.5 * dx, "lattice": "grid",
                    "calibrate_volume": True, "speed_of_sound": arguments.sound_ratio * speed, "power": POWER,
                    "cfl": 0.15, "gravity": [0.0, 0.0, 0.0],
                    "background_pressure": arguments.background_ratio * dynamic_pressure},
        "numerics": {"use_density_diffusion": True, "delta_coefficient": 0.1, "use_kcg_correction": True,
                     "regularization": {"xi": 0.01, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                     "use_pst": not arguments.no_pst, "pst_main": arguments.pst_main, "pst_anti": 0.0005,
                     "defrag_enabled": True, "defrag_cadence": 10, "use_prefix_sum_defrag": False,
                     "solid_pressure": "accumulate", "solid_reaction_force": True,
                     "density_diffusion_gradient_term": False,
                     "pair_correction": "reverse",
                     "pst_near_solid": arguments.pst_near_solid,
                     "thin_plate_dashpot": 0.0, "thin_plate_viscosity": True},
        "capacities": {"pool_size": pool, "max_per_voxel": 64, "max_incoming": 32, "workgroup": 128},
        "material_library": "materials.yaml",
        "geometry": {"frame": "frame.obj",
                     "particles": [{"file": "fluid.obj", "material": "box_fluid"},
                                   {"file": "wall.obj", "material": "box_wall"},
                                   {"file": "plate.obj", "material": "plate", "thin_plate": "paddle"}]},
        "rotor": {"axis": [0.0, 0.0, 1.0], "pivot": [-radius, 0.0, 0.0], "ramp_time": arguments.ramp * arguments.plate_length / speed},
        "thin_plates": [{"name": "paddle", "shape": "rectangle", "frame": "rotor",
                         "centre": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0], "axis_a": [1.0, 0.0, 0.0],
                         # outline of the pair cut-off: the particle line plus --edge-extension * h beyond each free
                         # edge (0 = the geometric plate, the production treatment)
                         "extent": [float(half_a_sites * dx + arguments.edge_extension * hdx * dx), float(dx)],
                         "thickness": float(arguments.thickness_ratio * arguments.plate_length),
                         "point_measure": float(dx)}],
    }
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": arguments.viscosity},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": arguments.viscosity},
                 "plate": {"kind": "rotor", "rest_density": REST_DENSITY, "viscosity": arguments.viscosity,
                           "rotor_angular_velocity": float(omega)}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    geometry = dict(dx=dx, half_length=half_length, radius=radius, omega=omega, t_end=t_end,
                    box=((xs[0], xs[-1]), (ys[0], ys[-1])), fluid_count=int(fluid.shape[0]),
                    plate_count=int(plate.shape[0]), wall_count=int(wall.shape[0]),
                    dynamic_pressure=dynamic_pressure)
    return directory / "case.yaml", geometry


def run_one(arguments, plate_cells, out):
    name = f"drag2d_L{plate_cells}{arguments.name_suffix}"
    case_path, geometry = build_case(out / name, arguments, plate_cells)
    case = load_case(str(case_path))
    dx = geometry["dx"]
    mass_factor = _calibrate_particle_volume(arguments.hdx * dx, 0.5 * dx, 2, "grid") / dx ** 2
    axis = np.array([0.0, 0.0, 1.0])
    pivot = np.array([-geometry["radius"], 0.0, 0.0])
    length = arguments.plate_length
    speed = arguments.speed
    q_length = geometry["dynamic_pressure"] * length                # 1/2 rho U^2 L per unit depth
    steps = int(math.ceil(geometry["t_end"] / case.timestep))
    report_every = max(1, int(round(arguments.report_interval * length / speed / case.timestep)))
    print(f"=== {name}: 2D, L/dx = {plate_cells}, dx {dx * 1e3:.3f} mm, h/dx {arguments.hdx}, fluid {geometry['fluid_count']:,}, "
          f"wall {geometry['wall_count']:,}, plate {geometry['plate_count']}, box x {geometry['box'][0][0] * 1e3:.0f}..{geometry['box'][0][1] * 1e3:.0f} "
          f"y {geometry['box'][1][0] * 1e3:.0f}..{geometry['box'][1][1] * 1e3:.0f} mm, U {speed} m/s, R/a {arguments.radius_ratio}, "
          f"Re {speed * length / arguments.viscosity:.0f}, c0 {case.physics.speed_of_sound} m/s, p_b {case.physics.background_pressure:.2f} Pa, "
          f"dt {case.timestep:.3e} s, {steps} steps to t* = {arguments.t_end}, report every {report_every} steps, mass factor {mass_factor:.4f}")
    history = []
    with VulkanContext.create(application_name="plate_drag_2d", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            while simulator.step_count < steps:
                simulator.step()
                if simulator.step_count % report_every != 0:
                    continue
                theta = simulator.rotor_angle
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                normal = np.array([-sin_t, cos_t, 0.0])             # y-axis of the plate frame turned by theta
                centre = pivot + np.array([geometry["radius"] * cos_t, geometry["radius"] * sin_t, 0.0])
                reactions = simulator.readback_thin_plate_reactions()
                force = reactions["force"] / mass_factor
                position = reactions["position"]
                f_normal = force @ normal
                f_tangent = force @ np.array([cos_t, sin_t, 0.0])
                side = (position - centre) @ normal                 # > 0: fluid ahead of the plate (pressure side)
                front = side > 0.0
                area = length                                       # per unit depth
                # drag opposes the motion (+n): C_D = -F_n / (q L). The fluid ahead pushes the plate back
                # (force -p_front A along n), the fluid behind pushes it forward (+p_back A): absolute
                # pressures, the background p_b is removed in the table.
                history.append((simulator.simulation_time, simulator.simulation_time * speed / length,
                                f_normal.sum(), f_tangent.sum(), f_normal[front].sum(), f_normal[~front].sum(),
                                int(front.sum()), int((~front).sum()),
                                -f_normal.sum() / q_length, -f_normal[front].sum() / area, f_normal[~front].sum() / area))
            status = simulator.readback_global_status()
            print(f"   end: alive {status['alive_particle_count']}, overflow {status['overflow_inside_count']}/"
                  f"{status['overflow_incoming_count']}, KCG fallback {status['correction_fallback_count']}")
        finally:
            simulator.destroy()
    history = np.asarray(history)
    header = "time,t_star,force_normal,force_tangent,force_front,force_back,records_front,records_back,cd,pressure_front,pressure_back"
    np.savetxt(out / f"{name}_history.csv", history, delimiter=",", header=header, comments="")
    t_star = history[:, 1]
    cd = history[:, 8]
    print("   t*      C_D (0.2 t* window mean)   C_D front   C_D back    C_p front   C_p back   KS ref   XN ref")
    rows = {}
    for t_ref in sorted(REFERENCE_KS_RE1000):
        if t_ref > arguments.t_end + 1e-9:
            continue
        window = (t_star >= t_ref - 0.1) & (t_star <= t_ref + 0.1)
        if window.sum() == 0:
            continue
        mean_cd = float(cd[window].mean())
        front = float(-history[window, 4].mean() / q_length)               # drag share of the pressure face
        back = float(-history[window, 5].mean() / q_length)                # drag share of the suction face
        p_b = case.physics.background_pressure
        p_front = float((history[window, 9].mean() - p_b) / geometry["dynamic_pressure"])   # C_p of the faces
        p_back = float((history[window, 10].mean() - p_b) / geometry["dynamic_pressure"])
        rows[t_ref] = (mean_cd, front, back, p_front, p_back)
        print(f"   {t_ref:4.1f}   {mean_cd:8.3f}                 {front:8.3f}   {back:8.3f}   {p_front:8.3f}   {p_back:8.3f}"
              f"   {REFERENCE_KS_RE1000[t_ref]:5.2f}   {REFERENCE_XN_RE1000.get(t_ref, float('nan')):5.2f}")
    return name, rows, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plate-cells", type=int, nargs="+", default=[10, 20, 40, 80], help="L / dx")
    parser.add_argument("--plate-length", type=float, default=0.02, help="L = 2a, m")
    parser.add_argument("--speed", type=float, default=0.05, help="U at the plate centre, m/s")
    parser.add_argument("--viscosity", type=float, default=1.0e-6, help="nu, m^2/s (Re = U L / nu)")
    parser.add_argument("--hdx", type=float, default=3.0)
    parser.add_argument("--sound-ratio", type=float, default=20.0, help="c0 / U")
    parser.add_argument("--background-ratio", type=float, default=4.0, help="p_b / (1/2 rho U^2)")
    parser.add_argument("--thickness-ratio", type=float, default=0.115, help="plate thickness / L")
    parser.add_argument("--radius-ratio", type=float, default=10.0, help="R / a of the pivot")
    parser.add_argument("--margin", type=float, default=3.0, help="box margin around the swept arc, in L")
    parser.add_argument("--ramp", type=float, default=0.05, help="spin-up time in units of L / U")
    parser.add_argument("--t-end", type=float, default=3.0, help="end time t* = t U / L")
    parser.add_argument("--report-interval", type=float, default=0.02, help="in units of L / U")
    parser.add_argument("--edge-extension", type=float, default=0.0,
                        help="extend the cut-off outline beyond the free edges by this many h (edge treatment test)")
    parser.add_argument("--pst-near-solid", default="full", choices=("full", "tangential"),
                        help="numerics.pst_near_solid: tangential removes the shift component along the solid normal")
    parser.add_argument("--pst-main", type=float, default=0.1)
    parser.add_argument("--no-pst", action="store_true")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--out", default="output/plate_drag_2d")
    arguments = parser.parse_args()
    out = ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    compile_shaders_v1.compile_v1_shaders()
    summary = {}
    for plate_cells in arguments.plate_cells:
        name, rows, geometry = run_one(arguments, plate_cells, out)
        summary[plate_cells] = rows
    print("\n   C_D at t* (window +- 0.1):   " + "   ".join(f"{t:4.1f}" for t in sorted(REFERENCE_KS_RE1000) if t <= arguments.t_end))
    for plate_cells, rows in summary.items():
        print(f"   L/dx = {plate_cells:4d}:                 " + "   ".join(f"{rows[t][0]:4.2f}" if t in rows else "  - "
                                                                         for t in sorted(REFERENCE_KS_RE1000) if t <= arguments.t_end))
    print("   KS 1996 (Re 1000):            " + "   ".join(f"{REFERENCE_KS_RE1000[t]:4.2f}" for t in sorted(REFERENCE_KS_RE1000) if t <= arguments.t_end))


if __name__ == "__main__":
    main()
