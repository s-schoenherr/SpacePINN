from __future__ import annotations

import numpy as np

from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem

from spacepinn.config.config_goddard import config_goddard
from spacepinn.config.config_orbit_transfer import GM_EARTH
from spacepinn.paper._rendezvous_hold_point_eci_shared import (
    TARGET_RADIUS_KM,
    TARGET_SPEED_KM_S,
    build_scenario,
)
from spacepinn.opengoddard._solve import solve_with_diagnostics
from spacepinn.opengoddard.legendre_patch import patch_opengoddard_legendre
from spacepinn.result import TrajectoryResult

DEFAULT_U_MAX = 0.05


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


def _cartesian_position_to_polar(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.linalg.norm(position, axis=1)
    alpha = np.unwrap(np.arctan2(position[:, 1], position[:, 0]))
    return rho, alpha


def _cartesian_vectors_to_polar_components(position: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.linalg.norm(position, axis=1) + 1e-15
    radial = np.sum(position * vector, axis=1) / rho
    tangential = (-position[:, 1] * vector[:, 0] + position[:, 0] * vector[:, 1]) / rho
    return radial, tangential


def _polar_to_cartesian_position(rho: float, alpha: float) -> np.ndarray:
    return np.asarray([rho * np.cos(alpha), rho * np.sin(alpha)], dtype=float)


def _cartesian_state_to_polar(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = float(np.linalg.norm(position))
    alpha = float(np.arctan2(position[1], position[0]))
    vr = float(np.dot(position, velocity) / (rho + 1e-15))
    vt = float((-position[1] * velocity[0] + position[0] * velocity[1]) / (rho + 1e-15))
    return np.asarray([rho, alpha], dtype=float), np.asarray([vr, vt], dtype=float)


class Spaceship:
    def __init__(self, *, time_final_guess: float):
        scenario = build_scenario(t_final_seconds=time_final_guess)
        self.m = 1.0
        self.r0_cart = np.asarray(scenario["chaser"]["start_position_km"], dtype=float)
        self.v0_cart = np.asarray(scenario["chaser"]["start_velocity_km_s"], dtype=float)
        self.r0, self.v0 = _cartesian_state_to_polar(self.r0_cart, self.v0_cart)
        self.target_radius = float(TARGET_RADIUS_KM)
        self.target_speed = float(TARGET_SPEED_KM_S)
        self.hold_point_radial_offset = float(scenario["chaser"]["final_hold_point_offset_km"][0])
        self.ao = np.asarray([[0.0, 0.0, GM_EARTH]], dtype=float)
        self.u_max = DEFAULT_U_MAX

    def compute_gravity_polar(self, rho):
        return -GM_EARTH / (rho + 1e-15) ** 2

    def target_state(self, time_final):
        mean_motion = np.sqrt(GM_EARTH / self.target_radius**3)
        theta = mean_motion * float(time_final)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        radial = np.asarray([cos_theta, sin_theta], dtype=float)
        along_track = np.asarray([-sin_theta, cos_theta], dtype=float)
        target_position = self.target_radius * radial
        hold_rho = self.target_radius + self.hold_point_radial_offset
        hold_position = hold_rho * radial
        target_velocity = self.target_radius * mean_motion * along_track
        return {
            "theta": theta,
            "mean_motion": mean_motion,
            "target_position": target_position,
            "hold_position": hold_position,
            "target_velocity": target_velocity,
            "hold_rho": hold_rho,
            "hold_alpha": theta,
            "hold_vr": 0.0,
            "hold_vt": hold_rho * mean_motion,
        }


def _cold_start_guess(obj: Spaceship, time_all_section: np.ndarray, *, time_final_guess: float) -> dict[str, np.ndarray]:
    terminal_state = obj.target_state(time_final_guess)
    time_grid = np.asarray(time_all_section, dtype=float)
    normalized_time = (time_grid - time_grid[0]) / (time_grid[-1] - time_grid[0] + 1e-15)

    # Use a small but structured control seed so the cold start is not effectively ballistic.
    tangential_thrust_guess = np.full_like(time_grid, 0.02 * obj.u_max, dtype=float)
    radial_thrust_guess = np.where(normalized_time <= 0.5, 0.01 * obj.u_max, -0.01 * obj.u_max)
    return {
        "rho": Guess.linear(time_grid, obj.r0[0], terminal_state["hold_rho"]),
        "alpha": Guess.linear(time_grid, obj.r0[1], terminal_state["hold_alpha"]),
        "vr": Guess.linear(time_grid, obj.v0[0], terminal_state["hold_vr"]),
        "vt": Guess.linear(time_grid, obj.v0[1], terminal_state["hold_vt"]),
        "ur": radial_thrust_guess,
        "ut": tangential_thrust_guess,
    }


def _warm_start_guess(result, target_time: np.ndarray) -> dict[str, np.ndarray]:
    source_time = _to_numpy(result.t).reshape(-1)
    if getattr(result, "coordinate_system", None) == "polar" and hasattr(result, "r_polar") and hasattr(result, "v_polar"):
        polar_state = _to_numpy(result.r_polar)
        polar_velocity = _to_numpy(result.v_polar)
        if hasattr(result, "F_rho") and hasattr(result, "F_alpha"):
            polar_control = np.column_stack([_to_numpy(result.F_rho).reshape(-1), _to_numpy(result.F_alpha).reshape(-1)])
        else:
            polar_control = np.zeros((len(source_time), 2), dtype=float)
    else:
        position = _to_numpy(result.r)
        velocity = _to_numpy(result.v)
        thrust = _to_numpy(result.F)
        rho, alpha = _cartesian_position_to_polar(position)
        vr, vt = _cartesian_vectors_to_polar_components(position, velocity)
        ur, ut = _cartesian_vectors_to_polar_components(position, thrust)
        polar_state = np.column_stack([rho, alpha])
        polar_velocity = np.column_stack([vr, vt])
        polar_control = np.column_stack([ur, ut])

    state_guess = _interpolate_columns(source_time, polar_state, target_time)
    velocity_guess = _interpolate_columns(source_time, polar_velocity, target_time)
    control_guess = _interpolate_columns(source_time, polar_control, target_time)
    return {
        "rho": state_guess[:, 0],
        "alpha": state_guess[:, 1],
        "vr": velocity_guess[:, 0],
        "vt": velocity_guess[:, 1],
        "ur": control_guess[:, 0],
        "ut": control_guess[:, 1],
    }


def kinematic_rendezvous_hold_point_eci_goddard(
    label: str = "Baseline (OpenGoddard)",
    *,
    warm_start_result=None,
    max_iteration: int = 100,
    ftol: float = 1e-11,
    slsqp_maxiter: int = 25,
    time_final_guess: float = 900.0,
    time_final_upper_bound: float | None = None,
):
    patch_opengoddard_legendre(Problem)
    obj = Spaceship(time_final_guess=float(time_final_guess))

    def dynamics(prob: Problem, obj: Spaceship, section):
        rho = prob.states(0, section)
        alpha = prob.states(1, section)
        vr = prob.states(2, section)
        vt = prob.states(3, section)
        ur = prob.controls(0, section)
        ut = prob.controls(1, section)

        gravity_rho = obj.compute_gravity_polar(rho)
        dx = Dynamics(prob, section)
        dx[0] = vr
        dx[1] = vt / (rho + 1e-15)
        dx[2] = vt**2 / (rho + 1e-15) + gravity_rho + ur / obj.m
        dx[3] = -vr * vt / (rho + 1e-15) + ut / obj.m
        return dx()

    def equality(prob: Problem, obj: Spaceship):
        rho = prob.states_all_section(0)
        alpha = prob.states_all_section(1)
        vr = prob.states_all_section(2)
        vt = prob.states_all_section(3)

        terminal = obj.target_state(prob.time_final(-1))

        result = Condition()
        result.equal(rho[0], obj.r0[0])
        result.equal(alpha[0], obj.r0[1])
        result.equal(rho[-1], terminal["hold_rho"])
        result.equal(alpha[-1], terminal["hold_alpha"])
        result.equal(vr[0], obj.v0[0])
        result.equal(vt[0], obj.v0[1])
        result.equal(vr[-1], terminal["hold_vr"])
        result.equal(vt[-1], terminal["hold_vt"])
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

    time_init = [0.0, float(time_final_guess)]
    n = [100]
    num_states = [4]
    num_controls = [2]
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    target_time = _target_time_grid(prob.time_all_section)
    guess = (
        _warm_start_guess(warm_start_result, target_time)
        if warm_start_result is not None
        else _cold_start_guess(obj, prob.time_all_section, time_final_guess=float(time_final_guess))
    )

    prob.set_states_all_section(0, guess["rho"])
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
        tf = prob.time_final(-1)
        terminal = obj.target_state(tf)
        print(f"tf: {tf:.5f}")
        print(f"hold_rho: {terminal['hold_rho']:.5f}")
        print(f"hold_alpha: {terminal['hold_alpha']:.5f}")

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
    alpha = prob.states_all_section(1)
    vr = prob.states_all_section(2)
    vt = prob.states_all_section(3)
    ur = prob.controls_all_section(0)
    ut = prob.controls_all_section(1)
    gravity_rho = obj.compute_gravity_polar(rho)
    gravity_polar = np.column_stack([gravity_rho, np.zeros_like(gravity_rho)])
    terminal = obj.target_state(t[-1])
    running_cost_values = np.asarray(running_cost(prob, obj), dtype=float)
    total_cost = float(np.trapezoid(running_cost_values, t))

    result = TrajectoryResult.from_open_goddard(
        label=label,
        t=t,
        r=np.column_stack([rho, alpha]),
        v=np.column_stack([vr, vt]),
        F=np.column_stack([ur, ut]),
        G=gravity_polar,
        r0=np.asarray([obj.r0[0], obj.r0[1]], dtype=float),
        rN=np.asarray([terminal["hold_rho"], terminal["hold_alpha"]], dtype=float),
        ao=obj.ao,
        coordinate_system="polar",
        total_cost=total_cost,
        runtime_seconds=runtime_seconds,
        solver=solver_metadata,
    )

    initial_guess_payload = {
        "mode": "pinn_warm_start" if warm_start_result is not None else "cold_start",
        "source_label": getattr(warm_start_result, "label", None) if warm_start_result is not None else None,
        "source_t_total": float(getattr(warm_start_result, "t_total")) if warm_start_result is not None else None,
        "states": "trajectory_seed" if warm_start_result is not None else "linear",
        "controls": "pinn_thrust_seed" if warm_start_result is not None else "constant",
    }

    return {
        "label": result.label,
        "result": result,
        **config_goddard["plotting"],
        "model": None,
        "config": {
            "backend": "OpenGoddard",
            "problem": "rendezvous_hold_point_eci_polar",
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
                "r0_cart": obj.r0_cart,
                "v0_cart": obj.v0_cart,
                "r0_polar": obj.r0,
                "v0_polar": obj.v0,
                "target_radius": obj.target_radius,
                "target_speed": obj.target_speed,
                "hold_point_radial_offset": obj.hold_point_radial_offset,
                "gravity_sources": obj.ao,
                "u_max": obj.u_max,
            },
            "initial_guess": initial_guess_payload,
            "constraints": {
                "hard_boundary_velocity": True,
                "hard_boundary_position": True,
                "terminal_state_depends_on_time_final": True,
            },
        },
    }
