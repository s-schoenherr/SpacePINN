# %%
import os
from pathlib import Path

import numpy as np

import spacepinn
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics

from spacepinn.config.config_goddard import config_goddard
from spacepinn.opengoddard._solve import solve_with_diagnostics
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.result import TrajectoryResult
from spacepinn.opengoddard.legendre_patch import patch_opengoddard_legendre
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
SWEEP_GROUPS = {
    "small": [0.33, 0.5, 0.66, 0.75, 0.9, 1.0],
    "medium": np.linspace(1.0, 1.1, 6).tolist(),
    "large": [1.0, 1.25, 1.5, 1.75, 2.0, 3.0],
}


def fixed_tof_3d_goddard(TOF=1.0, color="#475569", linestyle="solid"):
    label = rf"OpenGoddard $T={TOF:.2f}\,\mathrm{{s}}$"
    patch_opengoddard_legendre(Problem)

    class Spaceship:
        def __init__(self):
            self.m = 1.0
            self.x0 = -1.0
            self.y0 = -1.0
            self.z0 = 0.0
            self.xf = 1.0
            self.yf = 1.0
            self.zf = 0.0
            self.u_max = 0.0
            self.ao = np.array(
                [[-0.5, -1.0, 0.0, 0.5], [-0.2, 0.4, 0.0, 1.0], [0.8, 0.3, 0.0, 0.5]]
            )  # astronomical objects
            self.r0 = np.array([self.x0, self.y0, self.z0])
            self.rf = np.array([self.xf, self.yf, self.zf])

        def compute_gravity_cartesian(self, x, y, z):
            """
            x, y, z: (N,) arrays of particle positions
            self.ao_rgm: (M, 4) array -> [x_ao, y_ao, z_a0, gravitational_constant]
            return
            G: (N, 3) Gravity array
            """

            # Construct r from x, y and z
            r = np.stack((x, y, z), axis=1)  # shape (N, 3)
            G = np.zeros_like(r)

            for ao in self.ao:
                ao_pos = ao[:3]  # (3,)
                G_const = ao[3]

                r_diff = r - ao_pos  # (N, 2)
                distances = np.linalg.norm(r_diff, axis=1) + 1e-15  # (N,)
                denominator = distances**3  # (N,)

                # Broadcast division over x, y and z components
                G -= G_const * r_diff / denominator[:, None]
            return G

    def dynamics(prob: Problem, obj: Spaceship, section):
        # State variables: 2D Postion and Velocity
        x = prob.states(0, section)
        y = prob.states(1, section)
        z = prob.states(2, section)

        vx = prob.states(3, section)
        vy = prob.states(4, section)
        vz = prob.states(5, section)

        # Control variables: 2D Thrust
        ux = prob.controls(0, section)
        uy = prob.controls(1, section)
        uz = prob.controls(2, section)

        G = obj.compute_gravity_cartesian(x, y, z)
        dx = Dynamics(prob, section)
        dx[0] = vx
        dx[1] = vy
        dx[2] = vz
        dx[3] = ux / obj.m + G[:, 0]
        dx[4] = uy / obj.m + G[:, 1]
        dx[5] = uz / obj.m + G[:, 2]

        return dx()

    def equality(prob: Problem, obj: Spaceship):
        x = prob.states_all_section(0)
        y = prob.states_all_section(1)
        z = prob.states_all_section(2)
        tf = prob.time_final(-1)

        result = Condition()

        # Initial / final conditions
        result.equal(x[0], obj.x0)
        result.equal(y[0], obj.y0)
        result.equal(z[0], obj.z0)
        result.equal(x[-1], obj.xf)
        result.equal(y[-1], obj.yf)
        result.equal(z[-1], obj.zf)
        result.equal(tf, TOF)  # Fixing total time

        return result()

    def inequality(prob: Problem, obj: Spaceship):
        tf = prob.time_final(-1)

        result = Condition()
        result.lower_bound(tf, 0.0)

        return result()

    # Mayer Cost
    def cost(prob: Problem, obj: Spaceship):
        return 0.0

    # Lagrange Cost
    def running_cost(prob: Problem, obj: Spaceship):
        ux = prob.controls_all_section(0)
        uy = prob.controls_all_section(1)
        uz = prob.controls_all_section(2)

        return ux**2 + uy**2 + uz**2

    # ---------------------------------------------

    # Initial TOF Guess
    time_init = [0.0, TOF]
    n = [100]  # Collocation Points
    num_states = [6]  # Number of state variables
    num_controls = [3]  # Number of control variables
    max_iteration = 5  # Number of max iterations
    slsqp_maxiter = 25
    ftol = 1e-12

    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Spaceship()

    opengoddard_config = {
        "backend": "OpenGoddard",
        "problem": "fixed_tof_3d",
        "solver": {
            "time_init": time_init,
            "n": n,
            "num_states": num_states,
            "num_controls": num_controls,
            "max_iteration": max_iteration,
            "slsqp_maxiter": slsqp_maxiter,
            "ftol": ftol,
        },
        "constraints": {"fixed_tof": TOF},
        "spaceship": {
            "mass": obj.m,
            "r0": obj.r0,
            "rN": obj.rf,
            "gravity_sources": obj.ao,
        },
        "initial_guess": {
            "x": {"kind": "linear", "start": obj.x0, "end": obj.xf},
            "y": {"kind": "linear", "start": obj.y0, "end": obj.yf},
            "z": {"kind": "constant", "value": 0.0},
            "vx": {"kind": "constant", "value": 1.0},
            "vy": {"kind": "constant", "value": 1.0},
            "vz": {"kind": "constant", "value": 0.0},
            "ux": {"kind": "constant", "value": 0.0},
            "uy": {"kind": "constant", "value": 0.0},
            "uz": {"kind": "constant", "value": 0.0},
        },
    }

    # Initial Guess for Trajectory: Straight line from r0 -> rf
    x_init = Guess.linear(prob.time_all_section, obj.x0, obj.xf)
    y_init = Guess.linear(prob.time_all_section, obj.y0, obj.yf)
    z_init = Guess.constant(prob.time_all_section, 0.0)

    # Initial Guess for Velocity: Constant mag sqrt(2)
    vx_init = Guess.constant(prob.time_all_section, 1.0)
    vy_init = Guess.constant(prob.time_all_section, 1.0)
    vz_init = Guess.constant(prob.time_all_section, 0.0)

    # Initial Guess Thrust: Constant 0
    ux_init = Guess.constant(prob.time_all_section, 0.0)
    uy_init = Guess.constant(prob.time_all_section, 0.0)
    uz_init = Guess.constant(prob.time_all_section, 0.0)

    # Set initial guesses
    prob.set_states_all_section(0, x_init)
    prob.set_states_all_section(1, y_init)
    prob.set_states_all_section(2, z_init)

    prob.set_states_all_section(3, vx_init)
    prob.set_states_all_section(4, vy_init)
    prob.set_states_all_section(5, vz_init)

    prob.set_controls_all_section(0, ux_init)
    prob.set_controls_all_section(1, uy_init)
    prob.set_controls_all_section(2, uz_init)

    # ---------------------------------------------
    prob.dynamics = [dynamics]
    prob.cost = cost
    prob.running_cost = running_cost
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        tf = prob.time_final(-1)
        print("tf: {0:.5f}".format(tf))

    runtime_seconds, solver_metadata = solve_with_diagnostics(
        prob,
        obj,
        display_func,
        ftol=ftol,
        maxiter=slsqp_maxiter,
        label=label,
    )

    # ========================
    # Post Process
    # ------------------------
    # Convert parameter vector to variable
    t = prob.time_update()

    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    z = prob.states_all_section(2)

    vx = prob.states_all_section(3)
    vy = prob.states_all_section(4)
    vz = prob.states_all_section(5)

    ux = prob.controls_all_section(0)
    uy = prob.controls_all_section(1)
    uz = prob.controls_all_section(2)

    Gxyz = obj.compute_gravity_cartesian(x, y, z)
    running_cost_values = np.asarray(running_cost(prob, obj), dtype=float)
    total_cost = float(cost(prob, obj) + np.trapezoid(running_cost_values, t))

    r = np.stack([x, y, z], axis=1)
    v = np.stack([vx, vy, vz], axis=1)
    F = np.stack([ux, uy, uz], axis=1)

    result = TrajectoryResult.from_open_goddard(
        label,
        t,
        r,
        v,
        F,
        Gxyz,
        obj.r0,
        obj.rf,
        obj.ao,
        "cartesian",
        total_cost=total_cost,
        runtime_seconds=runtime_seconds,
        solver=solver_metadata,
    )

    return {
        "label": label,
        "result": result,
        "color": color,
        "linestyle": linestyle,
        "quiver_scale": config_goddard["plotting"].get("quiver_scale", 20),
        "model": None,
        "config": opengoddard_config,
    }


def _build_additional_entry(result_entry: dict) -> dict:
    return {
        "label": result_entry["label"],
        "result": result_entry["result"],
        "model": result_entry.get("model"),
        "config": result_entry.get("config"),
        "plotting": {key: result_entry[key] for key in ("linestyle", "color", "quiver_scale") if key in result_entry},
        "source": "opengoddard",
    }


def main(*, skip_plots: bool = False, print_summary: bool = True):
    sweep_groups = SWEEP_GROUPS
    if os.getenv("FAST_SMOKE", "0") == "1":
        sweep_groups = {name: values[:2] for name, values in SWEEP_GROUPS.items()}

    collection_runs = {}
    colors = [None, "#475569", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"]

    for group_name, t_total_values in sweep_groups.items():
        additional_entries = []
        for index, t_total in enumerate(t_total_values):
            color = colors[index] if index < len(colors) else None
            result_entry = fixed_TOF_goddard(TOF=t_total, color=color)
            additional_entries.append(_build_additional_entry(result_entry))

        collection_run = run_experiment_collection(
            configs=[],
            label=f"fixed_tof_goddard_{group_name}",
            additional_entries=additional_entries,
            run_root=str(RUN_ROOT),
        )
        collection_runs[group_name] = collection_run

        if print_summary:
            print_collection_run_summary(collection_run)

        if not skip_plots:
            plotter = TrajectoryPlotter(
                collection_run["entries"],
                dim=3,
                figsize=(7, 7),
                fig_prefix=f"fixed_tof_goddard_{group_name}",
                output_dir=collection_run["plot_output_dir"],
            )
            plotter.plot_all(plot_quiver=False)

    return collection_runs
