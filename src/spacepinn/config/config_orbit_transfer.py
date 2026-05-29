from enum import Enum
from functools import partial
from typing import Literal

import numpy as np
import torch

from .transform_functions import kinematic_polar_fn

GM_EARTH = 398600.0  # km^3/s^2
R_EARTH = 6378.0  # km
H_LEO = 500.0  # km
H_HEO = 2000.0  # km
H_GEO = 35786.0  # km

R_LEO = H_LEO + R_EARTH
R_HEO = H_HEO + R_EARTH
R_GEO = H_GEO + R_EARTH


class Orbit(Enum):
    LEO = H_LEO
    HEO = H_HEO
    GEO = H_GEO

    def __init__(self, altitude):
        self.h = altitude
        self.R = R_EARTH + altitude
        self.V = np.sqrt(GM_EARTH / self.R)


class OrbitalTransferBC:
    def __init__(
        self,
        orbit_from: Orbit,
        orbit_to: Orbit,
        alpha_0: float = 0,
        alpha_T: float = np.pi,
        m0: float = 2000.0,
        coordinate_system: Literal["polar", "cartesian"] = "polar",
    ):
        del m0
        tensor_dtype = torch.float64
        if coordinate_system == "polar":
            self.x0 = torch.tensor([orbit_from.R, alpha_0], dtype=tensor_dtype)
            self.xN = torch.tensor([orbit_to.R, alpha_T], dtype=tensor_dtype)
            self.v0 = torch.tensor([0, orbit_from.V], dtype=tensor_dtype)
            self.vN = torch.tensor([0, orbit_to.V], dtype=tensor_dtype)
        elif coordinate_system == "cartesian":
            self.x0 = torch.tensor(
                [orbit_from.R * np.cos(alpha_0), orbit_from.R * np.sin(alpha_0)],
                dtype=tensor_dtype,
            )
            self.xN = torch.tensor(
                [orbit_to.R * np.cos(alpha_T), orbit_to.R * np.sin(alpha_T)],
                dtype=tensor_dtype,
            )
            self.v0 = torch.tensor(
                [-orbit_from.V * np.sin(alpha_0), orbit_from.V * np.cos(alpha_0)],
                dtype=tensor_dtype,
            )
            self.vN = torch.tensor(
                [-orbit_to.V * np.sin(alpha_T), orbit_to.V * np.cos(alpha_T)],
                dtype=tensor_dtype,
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system!r}")

        self.T_hohnmann = self.hohnmann_transfer_time(orbit_from.R, orbit_to.R)
        self.T_from = self.orbit_loop_time(orbit_from.R)
        self.T_to = self.orbit_loop_time(orbit_to.R)

    def hohnmann_transfer_time(self, r0, rN):
        return np.pi * np.sqrt((r0 + rN) ** 3 / (8 * GM_EARTH))

    def orbit_loop_time(self, r):
        return 2 * np.pi * np.sqrt(r**3 / GM_EARTH)


leo_heo = OrbitalTransferBC(Orbit.LEO, Orbit.HEO)

circular_ot_kinematic_polar_config = {
    "label": "Kinematic tPINN",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(leo_heo.T_hohnmann))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            kinematic_polar_fn,
            x0=leo_heo.x0,
            xN=leo_heo.xN,
            v0=leo_heo.v0,
            vN=leo_heo.vN,
            transform_only_R=False,
        ),
    },
    "optimizer": {
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],
        "t_colloc": torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_heo.T_hohnmann).float(),
        "r0": leo_heo.x0,
        "rN": leo_heo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.01),
        "n_adam": 5000,
        "n_lbfgs": 0,
        "w_physics": 1.0,
        "w_bc": 0,
        "convergence_threshold": 1e-6,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#2ca02c",
        "quiver_scale": 1 / 250,
    },
}
