from functools import partial
import torch
from .transform_functions import position_fn, kinematic_fn
from ..plotting.style import PALETTE, plotting_style
from .shared_parameters import (
    x0_3d,
    xN_3d,
    v0_3d,
    vN_3d,
    ao_3d,
    t_colloc,
    t_total,
)

geometric_3d_config = {
    "label": "Geometric tPINN",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(position_fn, x0=x0_3d, xN=xN_3d),
    },
    "optimizer": {
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 2000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 0,
        "convergence_threshold": 1e-6,
    },
    "plotting": plotting_style(color=PALETTE["position"], linestyle="solid", quiver_scale=20),
}

ordinary_3d_config = {
    "label": "oPINN",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": None,
    },
    "optimizer": {
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 2_000,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 3.5,
        "convergence_threshold": 1e-6,
    },
    "plotting": plotting_style(color=PALETTE["vanilla"], linestyle="dashed"),
}

kinematic_3d_config = {
    "label": "Kinematic tPINN",
    "seed": 2809,
    "extra_parameters": {"t_total": torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))},
    "pinn": {
        "N_INPUT": 1,
        "N_OUTPUT": 3,
        "N_NEURONS": 50,
        "N_LAYERS": 3,
        "input_transform_fn": None,
        "output_transform_fn": partial(kinematic_fn, x0=x0_3d, xN=xN_3d, v0=v0_3d, vN=vN_3d),
    },
    "optimizer": {
        "ao_rgm": ao_3d,
        "t_colloc": t_colloc,
        "t_total": t_total,
        "r0": x0_3d,
        "rN": xN_3d,
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=10, lr=0.1),
        "n_adam": 0,
        "n_lbfgs": 10_000,
        "w_physics": 1.0,
        "w_bc": 0,
        "convergence_threshold": 1e-6,
    },
    "plotting": plotting_style(color=PALETTE["kinematic"], linestyle="dashdot"),
}
