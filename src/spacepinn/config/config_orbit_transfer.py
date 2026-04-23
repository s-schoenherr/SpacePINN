from enum import Enum
from typing import Literal
from functools import partial
import numpy as np
import torch

from .transform_functions import (
    position_fn,
    kinematic_fn,
    geometric_polar_fn,
    kinematic_polar_fn,
    kinematic_polar_polar_transform_only_vr,
)

GM_EARTH = 398600.0  # km^3/s^2
R_EARTH = 6378.0  # km
H_LEO = 500.0  # km
H_HEO = 2000.0  # km
H_GEO = 35786.0  # km

R_LEO = H_LEO + R_EARTH
R_HEO = H_HEO + R_EARTH
R_GEO = H_GEO + R_EARTH


phi_trainable = torch.nn.Parameter(torch.tensor(torch.pi, dtype=torch.float64).requires_grad_(True))


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
                [
                    -orbit_from.V * np.sin(alpha_0),
                    orbit_from.V * np.cos(alpha_0),
                ],
                dtype=tensor_dtype,
            )
            self.vN = torch.tensor(
                [
                    -orbit_to.V * np.sin(alpha_T),
                    orbit_to.V * np.cos(alpha_T),
                ],
                dtype=tensor_dtype,
            )
        self.T_hohnmann = self.hohnmann_transfer_time(orbit_from.R, orbit_to.R)
        self.T_from = self.orbit_loop_time(orbit_from.R)
        self.T_to = self.orbit_loop_time(orbit_to.R)

    def hohnmann_transfer_time(self, r0, rN):
        return np.pi * np.sqrt((r0 + rN) ** 3 / (8 * GM_EARTH))

    def orbit_loop_time(self, r):
        return 2 * np.pi * np.sqrt(r**3 / GM_EARTH)


# Cartesian ------------------------------------------------------------------------
# t_colloc = (0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, 1000)))).detach().view(-1, 1).requires_grad_(True),

# LEO to HEO inital conditions in cartesian
leo_heo_cart = OrbitalTransferBC(Orbit.LEO, Orbit.HEO, coordinate_system="cartesian")
circular_ot_geometric_cart_config = {
    "label": "Orbit-transfer",
    "seed": 2809,
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(position_fn, x0=leo_heo_cart.x0, xN=leo_heo_cart.xN),
    },
    "optimizer": {
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True),  #
        "t_total": torch.tensor(3600 * 5).float(),
        "r0": leo_heo_cart.xN,
        "rN": leo_heo_cart.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-2),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.01),
        "n_adam": 1000,
        "n_lbfgs": 10000,
        "w_physics": 1,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#1f77b4",
    },
}

circular_ot_kinematic_cart_config = {
    "label": "Orbit-transfer",
    "seed": 2809,
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            kinematic_fn,
            x0=leo_heo_cart.x0,
            xN=leo_heo_cart.xN,
            v0=leo_heo_cart.v0,
            vN=leo_heo_cart.vN,
        ),
    },
    "optimizer": {
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(3600 * 5).float(),
        "r0": leo_heo_cart.x0,
        "rN": leo_heo_cart.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-2),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.01),
        "n_adam": 1000,
        "n_lbfgs": 10000,
        "w_physics": 1,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#2ca02c",
    },
}

# Polar --------------------------------------------------------------------------
# Sanity Check if polar coordinates implementation. Config to travel half of the LEO circular orbit.
leo_sanity_check = OrbitalTransferBC(Orbit.LEO, Orbit.LEO, coordinate_system="polar")
polar_sanity_check_config = {
    "label": "Orbit-transfer",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(leo_sanity_check.T_from / 2))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(geometric_polar_fn, x0=leo_sanity_check.x0, xN=leo_sanity_check.xN),
    },
    "optimizer": {
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_sanity_check.T_from / 2).float(),
        "r0": leo_sanity_check.x0,
        "rN": leo_sanity_check.x0,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=100, lr=0.1),
        "n_adam": 1000,
        "n_lbfgs": 1000,
        "w_physics": 1.0,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#1f77b4",
    },
}


# LEO to HEO
leo_heo = OrbitalTransferBC(Orbit.LEO, Orbit.HEO)
circular_ot_geometric_polar_config = {
    "label": "Geometric tPINN",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(leo_heo.T_hohnmann))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(geometric_polar_fn, x0=leo_heo.x0, xN=leo_heo.xN),
    },
    "optimizer": {
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_heo.T_hohnmann).float(),
        "r0": leo_heo.x0,
        "rN": leo_heo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.01),
        "n_adam": 1000,
        "n_lbfgs": 0,
        "w_physics": 1.0,
        "w_bc": 0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#1f77b4",
        "quiver_scale": 1 / 250,
    },
}

circular_ot_vanilla_polar_config = {
    "label": "Orbit-transfer",
    "seed": 2809,
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": None,
    },
    "optimizer": {
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_heo.T_hohnmann).float(),
        "r0": leo_heo.x0,
        "rN": leo_heo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.01),
        "n_adam": 1000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 1.0,
    },
    "plotting": {
        "linestyle": "dashdot",
        "color": "#ff7f0e",
    },
}

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
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
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
    },
}

kinematic_only_vr_config = {
    "label": "Only R and VR constrained",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(leo_heo.T_hohnmann))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            kinematic_polar_polar_transform_only_vr,
            x0=leo_heo.x0,
            xN=leo_heo.xN,
            v0=leo_heo.v0,
            vN=leo_heo.vN,
        ),
    },
    "optimizer": {
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_heo.T_hohnmann).float(),
        "r0": leo_heo.x0,
        "rN": leo_heo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.01),
        "n_adam": 3000,
        "n_lbfgs": 0,
        "w_physics": 1.0,
        "w_bc": 0,
        "convergence_threshold": 1e-6,
    },
    "plotting": {"linestyle": "dashdot", "color": "#2ca02c", "quiver_scale": 1 / 200},
}
