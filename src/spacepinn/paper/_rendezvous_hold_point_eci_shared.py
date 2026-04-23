from __future__ import annotations

import numpy as np

from spacepinn.config.config_orbit_transfer import GM_EARTH, R_LEO

TARGET_RADIUS_KM = float(R_LEO)
TARGET_MEAN_MOTION_RAD_S = np.sqrt(GM_EARTH / TARGET_RADIUS_KM**3)
TARGET_SPEED_KM_S = TARGET_RADIUS_KM * TARGET_MEAN_MOTION_RAD_S
INITIAL_RELATIVE_OFFSET_KM = np.array([-1.0, -0.3], dtype=float)
FINAL_HOLD_POINT_OFFSET_KM = np.array([0.03, 0.0], dtype=float)
DEFAULT_T_FINAL_SECONDS = 900.0


def target_state_eci(*, t_seconds: float, radius_km: float = TARGET_RADIUS_KM, speed_km_s: float = TARGET_SPEED_KM_S):
    mean_motion = np.sqrt(GM_EARTH / radius_km**3)
    theta = mean_motion * float(t_seconds)
    position = np.array([radius_km * np.cos(theta), radius_km * np.sin(theta)], dtype=float)
    velocity = np.array([-speed_km_s * np.sin(theta), speed_km_s * np.cos(theta)], dtype=float)
    radial_unit = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    along_track_unit = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    return {
        "theta": theta,
        "mean_motion": mean_motion,
        "position": position,
        "velocity": velocity,
        "radial_unit": radial_unit,
        "along_track_unit": along_track_unit,
    }


def cartesian_state_to_polar(*, position_km, velocity_km_s):
    position = np.asarray(position_km, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    rho = float(np.linalg.norm(position))
    alpha = float(np.arctan2(position[1], position[0]))
    vr = float(np.dot(position, velocity) / (rho + 1e-15))
    vt = float((-position[1] * velocity[0] + position[0] * velocity[1]) / (rho + 1e-15))
    return {
        "position": np.array([rho, alpha], dtype=float),
        "velocity": np.array([vr, vt], dtype=float),
    }


def build_scenario(*, t_final_seconds: float = DEFAULT_T_FINAL_SECONDS) -> dict:
    target_start = target_state_eci(t_seconds=0.0)
    target_end = target_state_eci(t_seconds=t_final_seconds)

    chaser_start_position = target_start["position"] + INITIAL_RELATIVE_OFFSET_KM
    chaser_start_velocity = target_start["velocity"].copy()
    chaser_end_position = target_end["position"] + FINAL_HOLD_POINT_OFFSET_KM[0] * target_end["radial_unit"]
    hold_point_radius_km = TARGET_RADIUS_KM + FINAL_HOLD_POINT_OFFSET_KM[0]
    chaser_end_velocity = hold_point_radius_km * target_end["mean_motion"] * target_end["along_track_unit"]
    chaser_start_polar = cartesian_state_to_polar(position_km=chaser_start_position, velocity_km_s=chaser_start_velocity)
    chaser_end_polar = cartesian_state_to_polar(position_km=chaser_end_position, velocity_km_s=chaser_end_velocity)

    return {
        "t_final_seconds": float(t_final_seconds),
        "target": {
            "radius_km": TARGET_RADIUS_KM,
            "speed_km_s": TARGET_SPEED_KM_S,
            "start": target_start,
            "end": target_end,
        },
        "chaser": {
            "initial_relative_offset_km": INITIAL_RELATIVE_OFFSET_KM.copy(),
            "final_hold_point_offset_km": FINAL_HOLD_POINT_OFFSET_KM.copy(),
            "start_position_km": chaser_start_position,
            "start_velocity_km_s": chaser_start_velocity,
            "start_position_polar": chaser_start_polar["position"],
            "start_velocity_polar": chaser_start_polar["velocity"],
            "end_position_km": chaser_end_position,
            "end_velocity_km_s": chaser_end_velocity,
            "end_position_polar": chaser_end_polar["position"],
            "end_velocity_polar": chaser_end_polar["velocity"],
        },
    }
