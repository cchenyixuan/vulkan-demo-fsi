"""_check_blade_leak_tracking.py — count the fluid particles that pass THROUGH the impeller
blades by following individual particles (uid) through closely spaced frames (2026-09-28).

The pair test of _check_thin_solid_penetration.py over-counts on single-layer blades
(particles resting in the notches of the staircase jitter across the mid-plane). Here a
passage is an unambiguous event in the frame rotating with the rotor:

    the particle is on one side of the blade (n <= -w), then stays inside the slab
    |n| < w while it is within the blade's in-plane extent shrunk by one spacing, and
    leaves the slab on the OTHER side (n >= +w).

n is the coordinate normal to the blade mid-plane, w the slab half width (--slab, in
spacings; 1 for a single-layer blade, 2.5 for the 3-layer blade with skin whose faces are
2 dx from the mid-plane). A particle that leaves the slab through its rim (around the
blade edge) or returns to the side it came from is not counted. Frames are --dense-every
steps apart (default 10 steps, about 0.7 ms, a displacement of at most 0.25 dx).

  run      <solver env>  _check_blade_leak_tracking.py run CASE.yaml --dense-start S --dense-steps N
               [--dense-every 10] --out FRAMES.npz [--torque-log CSV] [--dump FINAL.npz]
           --torque-log: torque with the per-impeller split every 1000 steps from the start of
           the run (same columns as _run_v1_headless.py); --dump: final positions, material,
           velocity and status, as _run_v1_headless.py --dump. With both, one run gives the
           power number, the mean speed and the leak count of its last dense window.
           --status-log: every 1000 steps the alive count, the overflow counters and the pressure
           of the fluid (whole tank and the Rushton zone r < 72 mm, y = 18.7 .. 58.5 mm): mean,
           1 % / 50 % / 99 % quantiles and the fraction with P < 0 (2026-09-29, background
           pressure and gravity runs). --dump then also holds density_pressure.
           --budget BUDGET.csv (needs numerics.solid_reaction_force): every 1000 steps the angular
           momentum budget of the fluid about the rotor axis (2026-09-29), all in N m, with the
           particle masses of the solver (mass factor included):
             fluid_torque   sum_fluid m (x - p) x a            what the forces do to the fluid
             shift_torque   sum_fluid m shift x v / dt         what the particle shift does
             rotor_reaction, wall_reaction                     torque of the fluid ON rotor / walls
             internal       fluid_torque + rotor_reaction + wall_reaction
                            = torque the fluid receives from fluid-fluid pairs (0 if they conserve)
             angular_momentum, kinetic_energy                  of the fluid
           --slice-dir DIR --slice-interval 0.1 [--slice-half-width 1.0] [--slice-heights ...]
           (2026-09-29): every --slice-interval seconds of simulated time the fluid particles of
           a vertical slab through the rotor axis (|z| < half width, in spacings) and of horizontal
           slabs at --slice-heights are written to DIR/slice_NNNN.npz: position, velocity and the
           vorticity component omega_z that force.comp computes (acceleration.w), time and rotor
           angle. Images: _plot_vorticity_slices.py.
  analyze  <python>      _check_blade_leak_tracking.py analyze FRAMES.npz --dx 0.003 --slab 1.0

The result is given as particles per second and as a mass flow, compared with the
impellers' pumping capacity rho N_Q N D^3 with N_Q = 0.75 (2.2 kg/s at 200 rpm, D = 0.096 m).
"""
import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "utils" / "geometry", pathlib.Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

SUBSET_RADIUS = 0.065
SUBSET_HEIGHTS = ((0.010, 0.065), (0.165, 0.225))
PUMPING_MASS_FLOW = 998.0 * 0.75 * (200.0 / 60.0) * 0.096 ** 3


def angular_momentum_budget(simulator):
    """Angular momentum budget of the fluid about the rotor axis, see the module docstring."""
    case = simulator.case
    axis = np.asarray(case.rotor.axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    pivot = np.asarray(case.rotor.pivot, dtype=np.float64)
    positions = simulator.readback_positions()
    material = simulator.readback_material()
    live = simulator.live_slot_mask(positions)
    fluid_groups = np.asarray([m.group_id for m in case.materials if m.kind == 0], dtype=np.uint32)
    fluid = live & np.isin(material, fluid_groups)
    velocity_mass = simulator.readback_velocity_mass()
    arm = positions[fluid, :3].astype(np.float64) - pivot
    velocity = velocity_mass[fluid, :3].astype(np.float64)
    mass = velocity_mass[fluid, 3].astype(np.float64)
    acceleration = simulator.readback_acceleration()[fluid, :3].astype(np.float64)
    shift = simulator.readback_shift()[fluid, :3].astype(np.float64)
    fluid_torque = float((mass * (np.cross(arm, acceleration) @ axis)).sum())
    shift_torque = float((mass * (np.cross(shift, velocity) @ axis)).sum() / float(case.timestep))
    rotor_reaction = float(simulator.readback_rotor_torque()["torque_axis"])
    wall_positions, wall_forces = simulator.readback_boundary_forces()
    wall_reaction = float((np.cross(wall_positions - pivot, wall_forces) @ axis).sum())
    return {
        "fluid_torque": fluid_torque, "shift_torque": shift_torque,
        "rotor_reaction": rotor_reaction, "wall_reaction": wall_reaction,
        "internal": fluid_torque + rotor_reaction + wall_reaction,
        "angular_momentum": float((mass * (np.cross(arm, velocity) @ axis)).sum()),
        "kinetic_energy": float(0.5 * (mass * (velocity ** 2).sum(axis=1)).sum()),
    }


BUDGET_COLUMNS = ("fluid_torque", "shift_torque", "rotor_reaction", "wall_reaction", "internal",
                  "angular_momentum", "kinetic_energy")


def run(arguments):
    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(arguments.case)
    frames = {}
    times, angles, steps = [], [], []
    with VulkanContext.create(application_name="blade_leak_tracking", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            end = arguments.dense_start + arguments.dense_steps
            torque_log = None
            if arguments.torque_log:
                torque_log = open(arguments.torque_log, "w")
                torque_log.write("step,time,angle,torque_axis,fx,fy,fz,torque_lower,torque_upper,torque_shaft,"
                                 "wall_torque,baffle_torque,wall_fx,wall_fy,wall_fz,torque_bell\n")
            slice_directory = None
            slice_index = 0
            if arguments.slice_dir:
                slice_directory = pathlib.Path(arguments.slice_dir)
                slice_directory.mkdir(parents=True, exist_ok=True)
                spacing = 2.0 * float(case.physics.particle_radius)
                slice_half_width = arguments.slice_half_width * spacing
            budget_log = None
            if arguments.budget:
                if not case.numerics.solid_reaction_force:
                    raise SystemExit("--budget needs numerics.solid_reaction_force")
                budget_log = open(arguments.budget, "w")
                budget_log.write("step,time," + ",".join(BUDGET_COLUMNS) + chr(10))
            status_log = None
            if arguments.status_log:
                status_log = open(arguments.status_log, "w")
                status_log.write("step,time,alive,overflow_inside,overflow_incoming,correction_fallback,"
                                 "p_mean,p_q01,p_q50,p_q99,p_negative_fraction,"
                                 "rushton_p_mean,rushton_p_q01,rushton_p_q50,rushton_p_q99,rushton_p_negative_fraction,"
                                 "density_mean,density_min,density_max\n")
            while simulator.step_count < end:
                simulator.step()
                step = simulator.step_count
                if slice_directory is not None and simulator.simulation_time >= slice_index * arguments.slice_interval:
                    positions = simulator.readback_positions()
                    fluid = simulator.live_slot_mask(positions) & (simulator.readback_material() == 0)
                    vertical = fluid & (np.abs(positions[:, 2]) < slice_half_width)
                    horizontal = np.zeros_like(fluid)
                    for height in arguments.slice_heights:
                        horizontal |= fluid & (np.abs(positions[:, 1] - height) < slice_half_width)
                    selected = vertical | horizontal
                    acceleration = simulator.readback_acceleration()
                    np.savez_compressed(
                        slice_directory / f"slice_{slice_index:04d}.npz",
                        position=positions[selected, :3].astype(np.float32),
                        velocity=simulator.readback_velocity_mass()[selected, :3].astype(np.float32),
                        vorticity_z=acceleration[selected, 3].astype(np.float32),
                        time=simulator.simulation_time, step=step, rotor_angle=simulator.rotor_angle,
                        spacing=spacing, half_width=slice_half_width,
                        heights=np.asarray(arguments.slice_heights, dtype=np.float64))
                    slice_index += 1
                if budget_log is not None and step % 1000 == 0:
                    budget = angular_momentum_budget(simulator)
                    budget_log.write(f"{step},{simulator.simulation_time:.6e},"
                                     + ",".join(f"{budget[name]:.6e}" for name in BUDGET_COLUMNS) + chr(10))
                    budget_log.flush()
                if status_log is not None and step % 1000 == 0:
                    status = simulator.readback_global_status()
                    positions = simulator.readback_positions()
                    fluid = simulator.live_slot_mask(positions) & (simulator.readback_material() == 0)
                    density_pressure = simulator.readback_density_pressure()
                    pressure = density_pressure[fluid, 1].astype(np.float64)
                    density = density_pressure[fluid, 0].astype(np.float64)
                    x = positions[fluid, :3]
                    zone = (np.hypot(x[:, 0], x[:, 2]) < 0.072) & (x[:, 1] > 0.0187) & (x[:, 1] < 0.0585)
                    fields = [step, f"{simulator.simulation_time:.6e}", status["alive_particle_count"],
                              status["overflow_inside_count"], status["overflow_incoming_count"],
                              status["correction_fallback_count"]]
                    for values in (pressure, pressure[zone]):
                        if values.size == 0:
                            fields += ["nan"] * 5
                            continue
                        q01, q50, q99 = np.percentile(values, [1, 50, 99])
                        fields += [f"{values.mean():.4e}", f"{q01:.4e}", f"{q50:.4e}", f"{q99:.4e}",
                                   f"{(values < 0).mean():.4f}"]
                    fields += [f"{density.mean():.6e}", f"{density.min():.6e}", f"{density.max():.6e}"]
                    status_log.write(",".join(str(value) for value in fields) + "\n")
                    status_log.flush()
                if torque_log is not None and step % 1000 == 0:
                    torque = simulator.readback_rotor_torque(0.1, 0.007)
                    fx, fy, fz = torque["force"]
                    torque_log.write(f"{step},{simulator.simulation_time:.6e},{torque['rotor_angle']:.6e},"
                                     f"{torque['torque_axis']:.6e},{fx:.6e},{fy:.6e},{fz:.6e},"
                                     f"{torque['torque_axis_lower']:.6e},{torque['torque_axis_upper']:.6e},"
                                     f"{torque['torque_axis_shaft']:.6e},")
                    if case.numerics.solid_reaction_force:
                        # torque of the fluid on the walls about the rotor axis (+y through the
                        # origin); "baffle": wall particles inside the tank radius above the floor
                        # thin plates (2026-09-30): the records of the static plates (the baffle
                        # plates) come with the index of their plate and count as baffle
                        wall_positions, wall_forces, wall_plate = simulator.readback_boundary_forces(
                            with_plate_index=True)
                        wall_torque = (wall_positions[:, 2] * wall_forces[:, 0]
                                       - wall_positions[:, 0] * wall_forces[:, 2])
                        wall_radius = np.hypot(wall_positions[:, 0], wall_positions[:, 2])
                        baffle = (((wall_radius < 0.1395) & (wall_radius > 0.10) & (wall_positions[:, 1] > 0.0))
                                  | (wall_plate >= 0))
                        total = wall_forces.sum(axis=0)
                        torque_log.write(f"{wall_torque.sum():.6e},{wall_torque[baffle].sum():.6e},"
                                         f"{total[0]:.6e},{total[1]:.6e},{total[2]:.6e},")
                    else:
                        torque_log.write("nan,nan,nan,nan,nan,")
                    # torque on the rotating bell below the Rushton hub (rotor particles below
                    # y = 20.5 mm outside the shaft radius); it is part of torque_lower
                    bell = simulator.readback_rotor_torque(0.0205, 0.007)
                    torque_log.write(f"{bell['torque_axis_lower']:.6e}\n")
                    torque_log.flush()
                if step >= arguments.dense_start and (step - arguments.dense_start) % arguments.dense_every == 0:
                    positions = simulator.readback_positions()
                    live = simulator.live_slot_mask(positions)
                    fluid = live & (simulator.readback_material() == 0)
                    radius = np.hypot(positions[:, 0], positions[:, 2])
                    height = positions[:, 1]
                    near = fluid & (radius < SUBSET_RADIUS) & (
                        ((height > SUBSET_HEIGHTS[0][0]) & (height < SUBSET_HEIGHTS[0][1]))
                        | ((height > SUBSET_HEIGHTS[1][0]) & (height < SUBSET_HEIGHTS[1][1])))
                    index = len(times)
                    frames[f"uid_{index}"] = simulator.readback_particle_uid()[near]
                    frames[f"pos_{index}"] = positions[near, :3].copy()
                    times.append(simulator.simulation_time); angles.append(simulator.rotor_angle); steps.append(step)
            status = simulator.readback_global_status()
            mass = float(simulator.readback_velocity_mass()[1, 3])
            if torque_log is not None:
                torque_log.close()
            if status_log is not None:
                status_log.close()
            if budget_log is not None:
                budget_log.close()
            if arguments.dump:
                import json
                np.savez(arguments.dump, positions=simulator.readback_positions(),
                         material=simulator.readback_material(), velocity_mass=simulator.readback_velocity_mass(),
                         particle_uid=simulator.readback_particle_uid(), status=json.dumps(status),
                         density_pressure=simulator.readback_density_pressure())
        finally:
            simulator.destroy()
    np.savez_compressed(arguments.out, times=np.array(times), angles=np.array(angles), steps=np.array(steps),
                        particle_mass=mass, **frames)
    print(f"[tracking] {len(times)} frames, t = {times[0]:.4f} .. {times[-1]:.4f} s, alive {status['alive_particle_count']:,}, "
          f"overflow {status['overflow_inside_count']}/{status['overflow_incoming_count']} -> {arguments.out}")


def analyze(arguments):
    from _check_thin_solid_penetration import rotate_back, sheets
    archive = np.load(arguments.frames)
    dx, slab = arguments.dx, arguments.slab * arguments.dx
    times, angles = archive["times"], archive["angles"]
    count = len(times)
    sheet_list, _, _ = sheets(dx)
    all_uid = np.unique(np.concatenate([archive[f"uid_{k}"] for k in range(count)]))
    duration = times[-1] - times[0]
    print(f"{arguments.frames}: {count} frames over {duration:.4f} s (t = {times[0]:.3f} .. {times[-1]:.3f} s), "
          f"{len(all_uid):,} particles tracked, slab half width {arguments.slab:g} dx")
    totals = {}
    for name, moving, centre, axis_a, axis_b, normal, half_a, half_b in sheet_list:
        if not moving:
            continue
        # state per frame: 0 absent / elsewhere, -1 / +1 on a side, 2 inside the slab (interior extent)
        state = np.zeros((count, len(all_uid)), dtype=np.int8)
        for k in range(count):
            local = rotate_back(archive[f"pos_{k}"].astype(np.float64), float(angles[k])) - centre
            a, b, n = local @ axis_a, local @ axis_b, local @ normal
            column = np.searchsorted(all_uid, archive[f"uid_{k}"])
            in_extent = (np.abs(a) < half_a + dx) & (np.abs(b) < half_b + dx)          # unshrunk extent
            interior = (np.abs(a) < half_a) & (np.abs(b) < half_b)                       # shrunk by dx (sheets())
            value = np.zeros(len(n), dtype=np.int8)
            side = in_extent & (np.abs(n) >= slab) & (np.abs(n) < 5 * dx)
            value[side & (n < 0)] = -1
            value[side & (n > 0)] = 1
            value[interior & (np.abs(n) < slab)] = 2
            state[k, column] = value
        candidates = np.nonzero((state == 2).any(axis=0))[0]
        forward = backward = 0
        for column in candidates:
            series = state[:, column]
            inside = series == 2
            edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
            for start, end in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
                if start == 0 or end >= count:
                    continue
                before, after = series[start - 1], series[end]
                if before * after == -1:                      # -1 -> +1 or +1 -> -1
                    if after == 1:
                        forward += 1
                    else:
                        backward += 1
        group = name.rsplit(" ", 1)[0]
        entry = totals.setdefault(group, [0, 0, 0])
        entry[0] += forward; entry[1] += backward; entry[2] += len(candidates)
    mass = float(archive["particle_mass"])
    for group, (forward, backward, candidates) in totals.items():
        passages = forward + backward
        print(f"   {group}s (6): passages {passages} (+n {forward}, -n {backward}) among {candidates} particles that entered a slab; "
              f"{passages / duration:.0f} particles/s = {passages / duration * mass * 1e3:.1f} g/s = "
              f"{passages / duration * mass / PUMPING_MASS_FLOW * 100:.2f} % of the pumping capacity {PUMPING_MASS_FLOW:.2f} kg/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="phase", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("case")
    run_parser.add_argument("--dense-start", type=int, required=True)
    run_parser.add_argument("--dense-steps", type=int, default=3000)
    run_parser.add_argument("--dense-every", type=int, default=10)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--torque-log", default=None)
    run_parser.add_argument("--dump", default=None)
    run_parser.add_argument("--status-log", default=None)
    run_parser.add_argument("--budget", default=None)
    run_parser.add_argument("--slice-dir", default=None)
    run_parser.add_argument("--slice-interval", type=float, default=0.1, help="seconds of simulated time")
    run_parser.add_argument("--slice-half-width", type=float, default=1.0, help="half thickness of the slabs, in spacings")
    run_parser.add_argument("--slice-heights", type=float, nargs="*", default=[0.0384, 0.120, 0.19465, 0.300],
                            help="heights y of the horizontal slabs (m)")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("frames")
    analyze_parser.add_argument("--dx", type=float, default=0.003)
    analyze_parser.add_argument("--slab", type=float, default=1.0, help="slab half width in spacings")
    arguments = parser.parse_args()
    run(arguments) if arguments.phase == "run" else analyze(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
