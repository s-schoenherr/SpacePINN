from __future__ import annotations

import numpy as np

from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem

from spacepinn.config.config_goddard import config_goddard
from spacepinn.config.config_orbit_transfer import GM_EARTH, OrbitalTransferBC, Orbit
from spacepinn.opengoddard.legendre_patch import patch_opengoddard_legendre
from spacepinn.opengoddard._solve import solve_with_diagnostics
from spacepinn.result import TrajectoryResult

leo_heo = OrbitalTransferBC(Orbit.LEO, Orbit.HEO, 0, np.pi, coordinate_system="polar")
DEFAULT_U_MAX = 0.01


class Spaceship:
    def __init__(self, transfer_bc: OrbitalTransferBC | None = None):
        transfer_bc = transfer_bc or leo_heo
        self.m = 1.0
        self.x0 = transfer_bc.x0.numpy()
        self.xf = transfer_bc.xN.numpy()
        self.v0 = transfer_bc.v0.numpy()
        self.vf = transfer_bc.vN.numpy()
        self.u_max = DEFAULT_U_MAX
        self.ao = np.array([[0.0, 0.0, GM_EARTH]])

    def compute_gravitiy_polar(self, r):
        return -GM_EARTH / (r + 1e-15) ** 2


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


def _warm_start_guess(result, target_time: np.ndarray) -> dict[str, np.ndarray]:
    if getattr(result, "coordinate_system", None) != "polar":
        raise ValueError("Warm start result must use polar coordinates.")

    source_time = _to_numpy(result.t).reshape(-1)
    polar_state = _to_numpy(getattr(result, "r_polar", result.r))
    polar_velocity = _to_numpy(getattr(result, "v_polar", result.v))
    thrust_rho = _to_numpy(getattr(result, "F_rho")).reshape(-1)
    thrust_alpha = _to_numpy(getattr(result, "F_alpha")).reshape(-1)

    state_guess = _interpolate_columns(source_time, polar_state, target_time)
    velocity_guess = _interpolate_columns(source_time, polar_velocity, target_time)
    control_guess = _interpolate_columns(source_time, np.column_stack([thrust_rho, thrust_alpha]), target_time)

    return {
        "r": state_guess[:, 0],
        "vr": velocity_guess[:, 0],
        "vt": velocity_guess[:, 1],
        "ur": control_guess[:, 0],
        "ut": control_guess[:, 1],
    }


def _cold_start_guess(obj: Spaceship, time_all_section: np.ndarray) -> dict[str, np.ndarray]:
    time_span = float(time_all_section[-1] - time_all_section[0] + 1e-15)
    tangential_thrust_guess = min(max(1e-6, abs(float(obj.vf[-1] - obj.v0[-1])) / time_span), obj.u_max)
    return {
        "r": Guess.linear(time_all_section, obj.x0[0], obj.xf[0]),
        "vr": Guess.linear(time_all_section, obj.v0[0], obj.vf[0]),
        "vt": Guess.linear(time_all_section, obj.v0[-1], obj.vf[-1]),
        "ur": Guess.constant(time_all_section, 0.0),
        "ut": Guess.constant(time_all_section, tangential_thrust_guess),
    }


def _integrate_alpha(time: np.ndarray, r: np.ndarray, vt: np.ndarray, alpha0: float) -> np.ndarray:
    angular_rate = vt / (r + 1e-15)
    alpha = np.empty_like(time, dtype=float)
    alpha[0] = float(alpha0)
    if len(time) == 1:
        return alpha
    dt = np.diff(time)
    increments = 0.5 * (angular_rate[:-1] + angular_rate[1:]) * dt
    alpha[1:] = alpha[0] + np.cumsum(increments)
    return alpha


def _solver_payload(
    *,
    label: str,
    obj: Spaceship,
    time_init: list[float],
    n: list[int],
    num_states: list[int],
    num_controls: list[int],
    max_iteration: int,
    ftol: float,
    slsqp_maxiter: int,
    time_final_upper_bound: float | None = None,
    warm_start_result=None,
) -> dict:
    initial_guess_payload = {"mode": "cold_start"}
    if warm_start_result is not None:
        initial_guess_payload = {
            "mode": "pinn_warm_start",
            "source_label": getattr(warm_start_result, "label", None),
            "source_coordinate_system": getattr(warm_start_result, "coordinate_system", None),
            "source_t_total": float(getattr(warm_start_result, "t_total")),
            "controls": "pinn_thrust_seed",
        }

    return {
        "backend": "OpenGoddard",
        "problem": "circular_orbit_transfer_polar_no_alpha",
        "variant": label,
        "solver": {
            "time_init": time_init,
            "n": n,
            "num_states": num_states,
            "num_controls": num_controls,
            "max_iteration": max_iteration,
            "slsqp_maxiter": slsqp_maxiter,
            "ftol": ftol,
            "time_final_upper_bound": time_final_upper_bound,
        },
        "spaceship": {
            "mass": obj.m,
            "r0": obj.x0,
            "rN": obj.xf,
            "v0": obj.v0,
            "vN": obj.vf,
            "u_max": obj.u_max,
            "gravity_sources": obj.ao,
        },
        "initial_guess": initial_guess_payload,
    }


def _orbit_transfer_goddard_no_alpha(
    *,
    label: str,
    warm_start_result=None,
    max_iteration: int = 5,
    ftol: float = 1e-12,
    slsqp_maxiter: int = 25,
    transfer_bc: OrbitalTransferBC | None = None,
    time_final_guess: float | None = None,
    time_final_upper_bound: float | None = None,
):
    patch_opengoddard_legendre(Problem)

    def dynamics(prob: Problem, obj: Spaceship, section):
        r = prob.states(0, section)
        vr = prob.states(1, section)
        vt = prob.states(2, section)
        ur = prob.controls(0, section)
        ut = prob.controls(1, section)

        gravity_rho = obj.compute_gravitiy_polar(r)
        dx = Dynamics(prob, section)
        dx[0] = vr
        dx[1] = vt**2 / (r + 1e-15) + gravity_rho + ur
        dx[2] = -vr * vt / (r + 1e-15) + ut
        return dx()

    def equality(prob: Problem, obj: Spaceship):
        r = prob.states_all_section(0)
        vr = prob.states_all_section(1)
        vt = prob.states_all_section(2)

        result = Condition()
        result.equal(r[0], obj.x0[0])
        result.equal(r[-1], obj.xf[0])
        result.equal(vr[0], obj.v0[0])
        result.equal(vr[-1], obj.vf[0])
        result.equal(vt[0], obj.v0[-1])
        result.equal(vt[-1], obj.vf[-1])
        return result()

    def inequality(prob: Problem, obj: Spaceship):
        del obj
        result = Condition()
        result.lower_bound(prob.time_final(-1), 0.0)
        if time_final_upper_bound is not None:
            result.upper_bound(prob.time_final(-1), float(time_final_upper_bound))
        return result()

    def cost(prob: Problem, obj: Spaceship):
        del prob, obj
        return 0.0

    def running_cost(prob: Problem, obj: Spaceship):
        del obj
        ur = prob.controls_all_section(0)
        ut = prob.controls_all_section(1)
        return ur**2 + ut**2

    transfer_bc = transfer_bc or leo_heo
    obj = Spaceship(transfer_bc)
    default_tof = transfer_bc.T_hohnmann
    solved_time_final_guess = (
        float(time_final_guess)
        if time_final_guess is not None
        else float(getattr(warm_start_result, "t_total", default_tof)) if warm_start_result else default_tof
    )

    time_init = [0.0, solved_time_final_guess]
    n = [100]
    num_states = [3]
    num_controls = [2]
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    target_time = _target_time_grid(prob.time_all_section)
    guess = _warm_start_guess(warm_start_result, target_time) if warm_start_result is not None else _cold_start_guess(
        obj,
        prob.time_all_section,
    )

    prob.set_states_all_section(0, guess["r"])
    prob.set_states_all_section(1, guess["vr"])
    prob.set_states_all_section(2, guess["vt"])
    prob.set_controls_all_section(0, guess["ur"])
    prob.set_controls_all_section(1, guess["ut"])

    prob.dynamics = [dynamics]
    prob.cost = cost
    prob.running_cost = running_cost
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        tf = prob.time_final(-1)
        print(f"tf: {tf:.5f}")

    runtime_seconds, solver_metadata = solve_with_diagnostics(
        prob,
        obj,
        display_func,
        ftol=ftol,
        maxiter=slsqp_maxiter,
        label=label,
    )

    t = prob.time_update()
    r = prob.states_all_section(0)
    vr = prob.states_all_section(1)
    vt = prob.states_all_section(2)
    ur = prob.controls_all_section(0)
    ut = prob.controls_all_section(1)
    alpha = _integrate_alpha(t, r, vt, alpha0=float(obj.x0[-1]))
    gravity_rho = obj.compute_gravitiy_polar(r)
    gravity_polar = np.column_stack((gravity_rho, np.zeros_like(gravity_rho)))
    running_cost_values = np.asarray(running_cost(prob, obj), dtype=float)
    total_cost = float(cost(prob, obj) + np.trapezoid(running_cost_values, t))

    result = TrajectoryResult.from_open_goddard(
        label=label,
        t=t,
        r=np.stack([r, alpha], axis=1),
        v=np.stack([vr, vt], axis=1),
        F=np.stack([ur, ut], axis=1),
        G=gravity_polar,
        r0=np.array([obj.x0[0], obj.x0[-1]]),
        rN=np.array([obj.xf[0], alpha[-1]]),
        ao=obj.ao,
        coordinate_system="polar",
        total_cost=total_cost,
        runtime_seconds=runtime_seconds,
        solver=solver_metadata,
    )

    return {
        "label": result.label,
        "result": result,
        **config_goddard["plotting"],
        "model": None,
        "config": _solver_payload(
            label=label,
            obj=obj,
            time_init=time_init,
            n=n,
            num_states=num_states,
            num_controls=num_controls,
            max_iteration=max_iteration,
            ftol=ftol,
            slsqp_maxiter=slsqp_maxiter,
            time_final_upper_bound=time_final_upper_bound,
            warm_start_result=warm_start_result,
        ),
    }


def kinematic_ot_goddard_free_final_angle_no_alpha(
    label="Direct collocation",
    *,
    warm_start_result=None,
    max_iteration: int = 5,
    ftol: float = 1e-12,
    slsqp_maxiter: int = 25,
    transfer_bc=None,
    time_final_guess: float | None = None,
    time_final_upper_bound: float | None = None,
):
    return _orbit_transfer_goddard_no_alpha(
        label=label,
        warm_start_result=warm_start_result,
        max_iteration=max_iteration,
        ftol=ftol,
        slsqp_maxiter=slsqp_maxiter,
        transfer_bc=transfer_bc,
        time_final_guess=time_final_guess,
        time_final_upper_bound=time_final_upper_bound,
    )
