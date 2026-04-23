from functools import partial
import torch
import numpy as np
from spacepinn.config.transform_functions import (
    mass_pinn_fn,
    mass_pinn_kinematic_fn,
)
from spacepinn.config.config_orbit_transfer import (
    OrbitalTransferBC,
    Orbit,
    GM_EARTH,
)

leo_heo = OrbitalTransferBC(Orbit.LEO, Orbit.HEO, alpha_T=np.pi, coordinate_system="polar")

Isp = 300
g0 = 9.81e-3

m0 = 500.0
mN = 100.0  # Make this trainable later


leo_heo_geometric_mass_pinn = {
    "label": "massPINN",
    "seed": 2809,
    "extra_parameters": {
        "t_total": torch.nn.Parameter(torch.tensor(leo_heo.T_hohnmann)),
        "mN": torch.nn.Parameter(torch.tensor(mN)),
    },
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2 + 1,  # Adding m here as output
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            mass_pinn_fn,
            x0=torch.cat([leo_heo.x0, torch.tensor([m0])]),
            xN=torch.cat([leo_heo.xN, torch.tensor([mN])]),
            transform_only_r=True,
            transform_only_m0=False,
        ),
    },
    "optimizer": {
        "massPINN": True,
        "dimensions": 2,
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_heo.T_hohnmann).float(),
        "r0": leo_heo.x0,
        "rN": leo_heo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-4),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.01),
        "n_adam": 10000,
        "n_lbfgs": 0,
        "w_physics": 1.0,
        "w_bc": 0,
        "kwargs": {"w_objective": 1e2, "Isp": Isp, "g0": g0},
        "convergence_threshold": 1e-5,
    },
    "plotting": {"linestyle": "dashdot", "color": "#1f77b4", "quiver_scale": 1 / 500},
}

leo_heo_kinematic_mass_pinn = {
    "label": "massPINN",
    "seed": 2809,
    "extra_parameters": {
        "t_total": torch.nn.Parameter(torch.tensor(leo_heo.T_hohnmann)),
    },
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2 + 1,  # Adding m here as output
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            mass_pinn_kinematic_fn,
            x0=torch.cat([leo_heo.x0, torch.tensor([m0])]),
            xN=torch.cat([leo_heo.xN, torch.tensor([mN])]),
            v0=leo_heo.v0,
            vN=leo_heo.vN,
        ),
    },
    "optimizer": {
        "massPINN": True,
        "dimensions": 2,
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 250).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_heo.T_hohnmann).float(),
        "r0": leo_heo.x0,
        "rN": leo_heo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-4),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.01),
        "n_adam": 10000,
        "n_lbfgs": 0,
        "w_physics": 1.0,
        "w_bc": 0,
        "convergence_threshold": 1e-6,
    },
    "plotting": {"linestyle": "dashdot", "color": "#2ca02c", "quiver_scale": 1 / 100},
}

leo_geo = OrbitalTransferBC(Orbit.LEO, Orbit.GEO, alpha_T=np.pi, coordinate_system="polar")
leo_geo_geometric_mass_pinn = {
    "label": "massPINN",
    "seed": 2809,
    "extra_parameters": {
        "t_total": torch.nn.Parameter(torch.tensor(leo_geo.T_hohnmann)),
    },
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 2 + 1,  # Adding m here as output
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(
            mass_pinn_fn,
            x0=torch.cat([leo_geo.x0, torch.tensor([3000])]),
            xN=torch.cat([leo_geo.xN, torch.tensor([mN])]),
            transform_only_r=False,
            transform_only_m0=False,
        ),
    },
    "optimizer": {
        "massPINN": True,
        "dimensions": 2,
        "coordinate_system": "polar",
        "ao_rgm": [[0, 0, GM_EARTH]],  # km^3/s^2
        "t_colloc": torch.linspace(0, 1, 500).view(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(leo_geo.T_hohnmann).float(),
        "r0": leo_geo.x0,
        "rN": leo_geo.xN,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.01),
        "n_adam": 10000,
        "n_lbfgs": 0,
        "w_physics": 1.0,
        "w_bc": 0,
        "convergence_threshold": 1e-6,
    },
    "plotting": {"linestyle": "dashdot", "color": "#1f77b4", "quiver_scale": 1 / 500},
}
