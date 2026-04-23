from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import numpy as np
import torch

from .config_orbit_transfer import GM_EARTH, Orbit, R_LEO

GM_MOON = 4902.800066  # km^3 / s^2
R_MOON = 1737.4  # km
EARTH_MOON_DISTANCE_KM = 384400.0  # km
EARTH_MOON_ORBITAL_PERIOD_SECONDS = 27.321661 * 86400.0
EARTH_MOON_MEAN_MOTION = 2.0 * np.pi / EARTH_MOON_ORBITAL_PERIOD_SECONDS
H_LLO = 100.0  # km
R_LLO = R_MOON + H_LLO
MOON_INITIAL_PHASE_RAD = 0.0


class MoonStateProvider(Protocol):
    def state_torch(
        self,
        t_seconds: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def state_numpy(self, t_seconds: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class CircularMoonStateProvider:
    radius_km: float = EARTH_MOON_DISTANCE_KM
    mean_motion_rad_s: float = EARTH_MOON_MEAN_MOTION
    phase0_rad: float = MOON_INITIAL_PHASE_RAD

    def state_torch(
        self,
        t_seconds: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_tensor = torch.as_tensor(t_seconds, dtype=dtype, device=device)
        theta = torch.as_tensor(self.phase0_rad, dtype=dtype, device=device) + (
            torch.as_tensor(self.mean_motion_rad_s, dtype=dtype, device=device) * t_tensor
        )
        radius = torch.as_tensor(self.radius_km, dtype=dtype, device=device)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        vx = -radius * torch.as_tensor(self.mean_motion_rad_s, dtype=dtype, device=device) * torch.sin(theta)
        vy = radius * torch.as_tensor(self.mean_motion_rad_s, dtype=dtype, device=device) * torch.cos(theta)
        if t_tensor.ndim == 0:
            return torch.stack((x, y)), torch.stack((vx, vy))
        return torch.cat((x, y), dim=-1), torch.cat((vx, vy), dim=-1)

    def state_numpy(self, t_seconds: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        t_array = np.asarray(t_seconds, dtype=float)
        theta = self.phase0_rad + self.mean_motion_rad_s * t_array
        x = self.radius_km * np.cos(theta)
        y = self.radius_km * np.sin(theta)
        vx = -self.radius_km * self.mean_motion_rad_s * np.sin(theta)
        vy = self.radius_km * self.mean_motion_rad_s * np.cos(theta)
        return np.stack((x, y), axis=-1), np.stack((vx, vy), axis=-1)


def llo_circular_speed(radius_km: float = R_LLO, mu_moon: float = GM_MOON) -> float:
    return float(np.sqrt(mu_moon / radius_km))


def translunar_time_guess_seconds(
    *,
    start_radius_km: float = R_LEO,
    target_radius_km: float = EARTH_MOON_DISTANCE_KM,
    time_guess_scale: float = 1.0,
) -> float:
    hohmann_like = np.pi * np.sqrt((start_radius_km + target_radius_km) ** 3 / (8.0 * GM_EARTH))
    return float(hohmann_like * time_guess_scale)


def leo_start_state_eci(*, phase_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    radius = Orbit.LEO.R
    speed = Orbit.LEO.V
    position = np.asarray([radius * np.cos(phase_rad), radius * np.sin(phase_rad)], dtype=float)
    velocity = np.asarray([-speed * np.sin(phase_rad), speed * np.cos(phase_rad)], dtype=float)
    return position, velocity


def moon_relative_target_state(
    *,
    t_seconds: float,
    alpha_rad: float,
    moon_provider: MoonStateProvider,
    llo_radius_km: float = R_LLO,
    llo_speed_km_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if llo_speed_km_s is None:
        llo_speed_km_s = llo_circular_speed(llo_radius_km)
    moon_position, moon_velocity = moon_provider.state_numpy(float(t_seconds))
    radial_unit = np.asarray([np.cos(alpha_rad), np.sin(alpha_rad)], dtype=float)
    tangential_unit = np.asarray([-np.sin(alpha_rad), np.cos(alpha_rad)], dtype=float)
    return (
        moon_position + llo_radius_km * radial_unit,
        moon_velocity + llo_speed_km_s * tangential_unit,
    )


def circular_moon_provider_config(provider: CircularMoonStateProvider) -> dict[str, float | str]:
    return {
        "kind": "circular",
        "radius_km": float(provider.radius_km),
        "mean_motion_rad_s": float(provider.mean_motion_rad_s),
        "phase0_rad": float(provider.phase0_rad),
    }


def build_moon_state_provider(config: dict[str, float | str] | None) -> MoonStateProvider:
    config = config or {"kind": "circular"}
    kind = str(config.get("kind", "circular")).lower()
    if kind != "circular":
        raise ValueError(f"Unsupported moon provider kind '{kind}'.")
    return CircularMoonStateProvider(
        radius_km=float(config.get("radius_km", EARTH_MOON_DISTANCE_KM)),
        mean_motion_rad_s=float(config.get("mean_motion_rad_s", EARTH_MOON_MEAN_MOTION)),
        phase0_rad=float(config.get("phase0_rad", MOON_INITIAL_PHASE_RAD)),
    )


def _moon_gravity_cartesian_acceleration(
    *,
    r,
    r_cart,
    v,
    a,
    t,
    t_total,
    moon_provider: MoonStateProvider,
    mu_moon: float = GM_MOON,
    eps: float = 1e-8,
):
    del r, v, a
    t_seconds = t * t_total
    moon_position, _ = moon_provider.state_torch(
        t_seconds,
        dtype=r_cart.dtype,
        device=r_cart.device,
    )
    displacement = r_cart - moon_position
    denominator = (torch.linalg.norm(displacement, dim=1, keepdim=True) + eps) ** 3
    return -torch.as_tensor(mu_moon, dtype=r_cart.dtype, device=r_cart.device) * displacement / denominator


def make_moon_gravity_acceleration_fn(
    moon_provider: MoonStateProvider,
    *,
    mu_moon: float = GM_MOON,
):
    return partial(
        _moon_gravity_cartesian_acceleration,
        moon_provider=moon_provider,
        mu_moon=mu_moon,
    )
