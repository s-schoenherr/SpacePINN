from __future__ import annotations

import numpy as np

from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem

from spacepinn.config.config_goddard import config_goddard
from spacepinn.config.shared_parameters import ao_3d, x0_3d, xN_3d
from spacepinn.opengoddard.legendre_patch import patch_opengoddard_legendre
from spacepinn.opengoddard._solve import solve_with_diagnostics
from spacepinn.result import TrajectoryResult


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _interpolate_columns(source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    if source_values.ndim == 1:
        return np.interp(target_time, source_time, source_values)
    return np.column_stack(
        [np.interp(target_time, source_time, source_values[:, column]) for column in range(source_values.shape[1])]
    )


def _target_time_grid(time_all_section: np.ndarray) -> np.ndarray:
    return (time_all_section - time_all_section[0]) / (time_all_section[-1] - time_all_section[0])


def _cold_start_guess(obj, time_all_section: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r": np.column_stack(
            [
                Guess.linear(time_all_section, obj.r0[0], obj.rf[0]),
                Guess.linear(time_all_section, obj.r0[1], obj.rf[1]),
                Guess.linear(time_all_section, obj.r0[2], obj.rf[2]),
            ]
        ),
        "v": np.column_stack(
            [
                Guess.linear(time_all_section, obj.v0[0], obj.vf[0]),
                Guess.linear(time_all_section, obj.v0[1], obj.vf[1]),
                Guess.linear(time_all_section, obj.v0[2], obj.vf[2]),
            ]
        ),
        "u": np.zeros((time_all_section.shape[0], 3), dtype=float),
    }


def _warm_start_guess(result, target_time: np.ndarray) -> dict[str, np.ndarray]:
    source_time = _to_numpy(result.t).reshape(-1)
    return {
        "r": _interpolate_columns(source_time, _to_numpy(result.r), target_time),
        "v": _interpolate_columns(source_time, _to_numpy(result.v), target_time),
        "u": _interpolate_columns(source_time, _to_numpy(result.F), target_time),
    }


def kinematic_sanity_check_3d_goddard(
    label: str = "Direct collocation",
    *,
    warm_start_result=None,
    max_iteration: int = 5,
):
    patch_opengoddard_legendre(Problem)

    class Spaceship:
        def __init__(self):
            self.m = 1.0
            self.r0 = _to_numpy(x0_3d).reshape(-1)
            self.rf = _to_numpy(xN_3d).reshape(-1)
            if warm_start_result is not None:
                self.v0 = _to_numpy(warm_start_result.v[0, :]).reshape(-1)
                self.vf = _to_numpy(warm_start_result.v[-1, :]).reshape(-1)
            else:
                self.v0 = np.zeros(3, dtype=float)
                self.vf = np.zeros(3, dtype=float)
            self.u_max = 0.0
            self.ao = np.asarray(ao_3d, dtype=float)

        def compute_gravity_cartesian(self, x, y, z):
            r = np.stack((x, y, z), axis=1)
            gravity = np.zeros_like(r)
            for ao in self.ao:
                ao_pos = ao[:3]
                g_const = ao[3]
                r_diff = r - ao_pos
                distances = np.linalg.norm(r_diff, axis=1) + 1e-15
                gravity -= g_const * r_diff / (distances**3)[:, None]
            return gravity

    def dynamics(prob: Problem, obj: Spaceship, section):
        x = prob.states(0, section)
        y = prob.states(1, section)
        z = prob.states(2, section)
        vx = prob.states(3, section)
        vy = prob.states(4, section)
        vz = prob.states(5, section)
        ux = prob.controls(0, section)
        uy = prob.controls(1, section)
        uz = prob.controls(2, section)

        gravity = obj.compute_gravity_cartesian(x, y, z)
        dx = Dynamics(prob, section)
        dx[0] = vx
        dx[1] = vy
        dx[2] = vz
        dx[3] = ux / obj.m + gravity[:, 0]
        dx[4] = uy / obj.m + gravity[:, 1]
        dx[5] = uz / obj.m + gravity[:, 2]
        return dx()

    def equality(prob: Problem, obj: Spaceship):
        x = prob.states_all_section(0)
        y = prob.states_all_section(1)
        z = prob.states_all_section(2)
        vx = prob.states_all_section(3)
        vy = prob.states_all_section(4)
        vz = prob.states_all_section(5)

        result = Condition()
        result.equal(x[0], obj.r0[0])
        result.equal(y[0], obj.r0[1])
        result.equal(z[0], obj.r0[2])
        result.equal(x[-1], obj.rf[0])
        result.equal(y[-1], obj.rf[1])
        result.equal(z[-1], obj.rf[2])
        result.equal(vx[0], obj.v0[0])
        result.equal(vy[0], obj.v0[1])
        result.equal(vz[0], obj.v0[2])
        result.equal(vx[-1], obj.vf[0])
        result.equal(vy[-1], obj.vf[1])
        result.equal(vz[-1], obj.vf[2])
        return result()

    def inequality(prob: Problem, obj: Spaceship):
        del obj
        tf = prob.time_final(-1)
        result = Condition()
        result.lower_bound(tf, 0.0)
        return result()

    def cost(prob: Problem, obj: Spaceship):
        del prob, obj
        return 0.0

    def running_cost(prob: Problem, obj: Spaceship):
        del obj
        ux = prob.controls_all_section(0)
        uy = prob.controls_all_section(1)
        uz = prob.controls_all_section(2)
        return ux**2 + uy**2 + uz**2

    obj = Spaceship()
    time_final_guess = float(getattr(warm_start_result, "t_total", 1.0))
    time_init = [0.0, time_final_guess]
    n = [100]
    num_states = [6]
    num_controls = [3]
    slsqp_maxiter = 25
    ftol = 1e-12

    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    target_time = _target_time_grid(prob.time_all_section)
    guess = (
        _warm_start_guess(warm_start_result, target_time)
        if warm_start_result is not None
        else _cold_start_guess(obj, prob.time_all_section)
    )

    for column in range(3):
        prob.set_states_all_section(column, guess["r"][:, column])
        prob.set_states_all_section(column + 3, guess["v"][:, column])
        prob.set_controls_all_section(column, guess["u"][:, column])

    prob.dynamics = [dynamics]
    prob.cost = cost
    prob.running_cost = running_cost
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        print(f"tf: {prob.time_final(-1):0.5f}")

    runtime_seconds, solver_metadata = solve_with_diagnostics(
        prob,
        obj,
        display_func,
        ftol=ftol,
        maxiter=slsqp_maxiter,
        label=label,
    )

    t = prob.time_update()
    r = np.column_stack([prob.states_all_section(0), prob.states_all_section(1), prob.states_all_section(2)])
    v = np.column_stack([prob.states_all_section(3), prob.states_all_section(4), prob.states_all_section(5)])
    F = np.column_stack([prob.controls_all_section(0), prob.controls_all_section(1), prob.controls_all_section(2)])
    Gxyz = obj.compute_gravity_cartesian(r[:, 0], r[:, 1], r[:, 2])
    running_cost_values = np.asarray(running_cost(prob, obj), dtype=float)
    total_cost = float(np.trapezoid(running_cost_values, t))

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

    initial_guess_payload = {
        "mode": "pinn_warm_start" if warm_start_result is not None else "cold_start",
        "source_label": getattr(warm_start_result, "label", None) if warm_start_result is not None else None,
        "source_t_total": float(getattr(warm_start_result, "t_total")) if warm_start_result is not None else None,
        "states": "trajectory_seed" if warm_start_result is not None else "linear",
        "controls": "pinn_thrust_seed" if warm_start_result is not None else "zero",
    }

    return {
        "label": result.label,
        "result": result,
        **config_goddard["plotting"],
        "model": None,
        "config": {
            "backend": "OpenGoddard",
            "problem": "kinematic_sanity_check_3d",
            "variant": label,
            "solver": {
                "time_init": time_init,
                "n": n,
                "num_states": num_states,
                "num_controls": num_controls,
                "max_iteration": max_iteration,
                "slsqp_maxiter": slsqp_maxiter,
                "ftol": ftol,
            },
            "spaceship": {
                "mass": obj.m,
                "r0": obj.r0,
                "rN": obj.rf,
                "v0": obj.v0,
                "vN": obj.vf,
                "gravity_sources": obj.ao,
            },
            "initial_guess": initial_guess_payload,
            "constraints": {
                "hard_boundary_velocity": True,
                "source": "Geometric tPINN",
            },
        },
    }
