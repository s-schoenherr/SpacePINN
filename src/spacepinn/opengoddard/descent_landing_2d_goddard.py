from __future__ import annotations

import numpy as np
from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem

from spacepinn.config.config_goddard import config_goddard
from spacepinn.config.config_orbit_transfer import GM_EARTH, Orbit, R_EARTH, R_LEO
from spacepinn.opengoddard._solve import solve_with_diagnostics
from spacepinn.opengoddard.legendre_patch import patch_opengoddard_legendre
from spacepinn.result import TrajectoryResult

DEFAULT_U_MAX = 0.05
ATMOSPHERE_BETA = 0.01
ATMOSPHERE_RHO0 = 1.225
ATMOSPHERE_SCALE_HEIGHT_KM = 7.2
ATMOSPHERE_LAYER_BREAK_KM = 50.0
ATMOSPHERE_UPPER_SCALE_HEIGHT_KM = 25.0


def _atmospheric_density(altitude_km: np.ndarray, *, drag_model: str) -> np.ndarray:
    model = drag_model.strip().lower()
    altitude = np.clip(np.asarray(altitude_km, dtype=float), 0.0, None)
    if model in {"none", "vacuum"}:
        return np.zeros_like(altitude)
    if model in {"exponential", "single_scale"}:
        return ATMOSPHERE_RHO0 * np.exp(-altitude / ATMOSPHERE_SCALE_HEIGHT_KM)
    if model in {"piecewise_exponential", "two_layer"}:
        break_density = ATMOSPHERE_RHO0 * np.exp(-ATMOSPHERE_LAYER_BREAK_KM / ATMOSPHERE_SCALE_HEIGHT_KM)
        lower_density = ATMOSPHERE_RHO0 * np.exp(-altitude / ATMOSPHERE_SCALE_HEIGHT_KM)
        upper_density = break_density * np.exp(-(altitude - ATMOSPHERE_LAYER_BREAK_KM) / ATMOSPHERE_UPPER_SCALE_HEIGHT_KM)
        return np.where(altitude <= ATMOSPHERE_LAYER_BREAK_KM, lower_density, upper_density)
    raise ValueError(f"Unsupported drag_model '{drag_model}'.")


class Spaceship:
    def __init__(self, *, alpha_final: float | None = None, drag_model: str = "none"):
        self.m = 1.0
        self.r0 = np.asarray([R_LEO, 0.0], dtype=float)
        self.v0 = np.asarray([0.0, Orbit.LEO.V], dtype=float)
        self.rf_rho = float(R_EARTH)
        self.vf = np.asarray([0.0, 0.0], dtype=float)
        self.alpha_final = None if alpha_final is None else float(alpha_final)
        self.drag_model = str(drag_model)
        self.u_max = DEFAULT_U_MAX
        self.ao = np.asarray([[0.0, 0.0, GM_EARTH]], dtype=float)

    def compute_gravity_polar(self, rho):
        return -GM_EARTH / (rho + 1e-15) ** 2

    def compute_drag_polar(self, rho, vr, vt):
        density = _atmospheric_density(rho - R_EARTH, drag_model=self.drag_model)
        speed_km_s = np.sqrt(vr**2 + vt**2 + 1e-12)
        speed_m_s = speed_km_s * 1000.0
        drag_m_s2 = 0.5 * density * ATMOSPHERE_BETA * speed_m_s**2
        drag_km_s2 = drag_m_s2 / 1000.0
        return -drag_km_s2 * (vr / speed_km_s), -drag_km_s2 * (vt / speed_km_s)


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


def _integrate_alpha(time: np.ndarray, rho: np.ndarray, vt: np.ndarray, alpha0: float) -> np.ndarray:
    angular_rate = vt / (rho + 1e-15)
    alpha = np.empty_like(time, dtype=float)
    alpha[0] = float(alpha0)
    if len(time) == 1:
        return alpha
    dt = np.diff(time)
    increments = 0.5 * (angular_rate[:-1] + angular_rate[1:]) * dt
    alpha[1:] = alpha[0] + np.cumsum(increments)
    return alpha


def _cold_start_guess(obj: Spaceship, time_all_section: np.ndarray) -> dict[str, np.ndarray]:
    time_span = float(time_all_section[-1] - time_all_section[0] + 1e-15)
    tangential_thrust_guess = min(max(1e-6, abs(float(obj.vf[-1] - obj.v0[-1])) / time_span), obj.u_max)
    radial_thrust_guess = np.where(
        np.linspace(0.0, 1.0, len(time_all_section)) <= 0.75,
        -0.01 * obj.u_max,
        0.01 * obj.u_max,
    )
    guess = {
        "rho": Guess.linear(time_all_section, obj.r0[0], obj.rf_rho),
        "vr": Guess.linear(time_all_section, obj.v0[0], obj.vf[0]),
        "vt": Guess.linear(time_all_section, obj.v0[1], obj.vf[1]),
        "ur": radial_thrust_guess,
        "ut": Guess.constant(time_all_section, -tangential_thrust_guess),
    }
    if obj.alpha_final is not None:
        guess["alpha"] = Guess.linear(time_all_section, obj.r0[1], obj.alpha_final)
    return guess


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
    guess = {
        "rho": state_guess[:, 0],
        "vr": velocity_guess[:, 0],
        "vt": velocity_guess[:, 1],
        "ur": control_guess[:, 0],
        "ut": control_guess[:, 1],
    }
    if state_guess.shape[1] > 1:
        guess["alpha"] = state_guess[:, 1]
    return guess


def kinematic_descent_landing_2d_goddard(
    label: str = "Baseline (OpenGoddard)",
    *,
    warm_start_result=None,
    max_iteration: int = 100,
    ftol: float = 1e-11,
    slsqp_maxiter: int = 25,
    time_final_guess: float = 1.0,
    time_final_upper_bound: float | None = None,
    alpha_final: float | None = None,
    drag_model: str = "none",
):
    patch_opengoddard_legendre(Problem)
    obj = Spaceship(alpha_final=alpha_final, drag_model=drag_model)

    def dynamics(prob: Problem, obj: Spaceship, section):
        rho = prob.states(0, section)
        ur = prob.controls(0, section)
        ut = prob.controls(1, section)
        if obj.alpha_final is None:
            vr = prob.states(1, section)
            vt = prob.states(2, section)
        else:
            vr = prob.states(2, section)
            vt = prob.states(3, section)

        gravity_rho = obj.compute_gravity_polar(rho)
        drag_rho, drag_t = obj.compute_drag_polar(rho, vr, vt)
        dx = Dynamics(prob, section)
        dx[0] = vr
        if obj.alpha_final is None:
            dx[1] = vt**2 / (rho + 1e-15) + gravity_rho + ur / obj.m + drag_rho
            dx[2] = -vr * vt / (rho + 1e-15) + ut / obj.m + drag_t
        else:
            dx[1] = vt / (rho + 1e-15)
            dx[2] = vt**2 / (rho + 1e-15) + gravity_rho + ur / obj.m + drag_rho
            dx[3] = -vr * vt / (rho + 1e-15) + ut / obj.m + drag_t
        return dx()

    def equality(prob: Problem, obj: Spaceship):
        rho = prob.states_all_section(0)
        if obj.alpha_final is None:
            vr = prob.states_all_section(1)
            vt = prob.states_all_section(2)
        else:
            alpha = prob.states_all_section(1)
            vr = prob.states_all_section(2)
            vt = prob.states_all_section(3)

        result = Condition()
        result.equal(rho[0], obj.r0[0])
        result.equal(rho[-1], obj.rf_rho)
        if obj.alpha_final is not None:
            result.equal(alpha[0], obj.r0[1])
            result.equal(alpha[-1], obj.alpha_final)
        result.equal(vr[0], obj.v0[0])
        result.equal(vr[-1], obj.vf[0])
        result.equal(vt[0], obj.v0[1])
        result.equal(vt[-1], obj.vf[1])
        return result()

    def inequality(prob: Problem, obj: Spaceship):
        rho = prob.states_all_section(0)
        ur = prob.controls_all_section(0)
        ut = prob.controls_all_section(1)
        thrust_mag = np.sqrt(ur**2 + ut**2)

        result = Condition()
        result.lower_bound(prob.time_final(-1), 0.0)
        if time_final_upper_bound is not None:
            result.upper_bound(prob.time_final(-1), float(time_final_upper_bound))
        result.lower_bound(rho, obj.rf_rho)
        result.upper_bound(thrust_mag, obj.u_max)
        return result()

    def cost(prob: Problem, obj: Spaceship):
        del prob, obj
        return 0.0

    def running_cost(prob: Problem, obj: Spaceship):
        del obj
        ur = prob.controls_all_section(0)
        ut = prob.controls_all_section(1)
        return ur**2 + ut**2

    solved_time_final_guess = (
        float(time_final_guess)
        if time_final_guess is not None
        else float(getattr(warm_start_result, "t_total", 1.0)) if warm_start_result is not None else 1.0
    )
    time_init = [0.0, solved_time_final_guess]
    n = [100]
    num_states = [3] if obj.alpha_final is None else [4]
    num_controls = [2]
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    target_time = _target_time_grid(prob.time_all_section)
    guess = _warm_start_guess(warm_start_result, target_time) if warm_start_result is not None else _cold_start_guess(
        obj,
        prob.time_all_section,
    )

    prob.set_states_all_section(0, guess["rho"])
    if obj.alpha_final is None:
        prob.set_states_all_section(1, guess["vr"])
        prob.set_states_all_section(2, guess["vt"])
    else:
        prob.set_states_all_section(1, guess["alpha"])
        prob.set_states_all_section(2, guess["vr"])
        prob.set_states_all_section(3, guess["vt"])
    prob.set_controls_all_section(0, guess["ur"])
    prob.set_controls_all_section(1, guess["ut"])

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
    rho = prob.states_all_section(0)
    if obj.alpha_final is None:
        vr = prob.states_all_section(1)
        vt = prob.states_all_section(2)
        alpha = _integrate_alpha(t, rho, vt, alpha0=float(obj.r0[-1]))
    else:
        alpha = prob.states_all_section(1)
        vr = prob.states_all_section(2)
        vt = prob.states_all_section(3)
    ur = prob.controls_all_section(0)
    ut = prob.controls_all_section(1)
    gravity_rho = obj.compute_gravity_polar(rho)
    gravity_polar = np.column_stack((gravity_rho, np.zeros_like(gravity_rho)))
    running_cost_values = np.asarray(running_cost(prob, obj), dtype=float)
    total_cost = float(cost(prob, obj) + np.trapezoid(running_cost_values, t))

    result = TrajectoryResult.from_open_goddard(
        label=label,
        t=t,
        r=np.stack([rho, alpha], axis=1),
        v=np.stack([vr, vt], axis=1),
        F=np.stack([ur, ut], axis=1),
        G=gravity_polar,
        r0=np.array([obj.r0[0], obj.r0[-1]]),
        rN=np.array([obj.rf_rho, alpha[-1]]),
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
        "config": {
            "backend": "OpenGoddard",
            "problem": "descent_landing_2d",
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
                "r0": obj.r0,
                "rho_final": obj.rf_rho,
                "v0": obj.v0,
                "vN": obj.vf,
                "u_max": obj.u_max,
                "gravity_sources": obj.ao,
                "drag_model": obj.drag_model,
            },
            "initial_guess": {
                "mode": "pinn_warm_start" if warm_start_result is not None else "cold_start",
                "source_label": getattr(warm_start_result, "label", None) if warm_start_result is not None else None,
            },
            "constraints": {
                "free_final_angle": alpha_final is None,
                "hard_terminal_velocity": True,
                "hard_terminal_radius": True,
                "fixed_terminal_angle": None if alpha_final is None else float(alpha_final),
            },
        },
        "source": "opengoddard",
    }
