from functools import partial

import pytest
import torch

from spacepinn.optimizer import TrajectoryOptimizer
from spacepinn.pinn import PINN
from spacepinn.runner.execution import execute_single_experiment


def _build_model():
    return PINN(
        N_INPUT=1,
        N_OUTPUT=2,
        N_NEURONS=8,
        N_LAYERS=2,
        input_transform_fn=None,
        output_transform_fn=None,
    )


def _base_optimizer_kwargs():
    return {
        "ao_rgm": [[0.0, 0.0, 1.0]],
        "t_colloc": torch.linspace(0.0, 1.0, 9, dtype=torch.float32).reshape(-1, 1).requires_grad_(True),
        "t_total": torch.tensor(1.0, dtype=torch.float32),
        "r0": torch.tensor([-1.0, -1.0], dtype=torch.float32),
        "rN": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "opt_adam": partial(torch.optim.Adam, lr=1e-3),
        "opt_lbfgs": partial(torch.optim.LBFGS, max_iter=1, lr=0.1),
        "n_adam": 1,
        "n_lbfgs": 0,
    }


def test_optimizer_requires_known_keys_only():
    kwargs = _base_optimizer_kwargs()
    kwargs["unknown_option"] = 123

    with pytest.raises(TypeError, match="Unknown optimizer kwargs"):
        TrajectoryOptimizer(_build_model(), **kwargs)


def test_optimizer_requires_all_mandatory_keys():
    kwargs = _base_optimizer_kwargs()
    kwargs.pop("rN")

    with pytest.raises(TypeError, match="Missing required optimizer kwargs"):
        TrajectoryOptimizer(_build_model(), **kwargs)


def test_optimizer_alias_mapping_uses_canonical_keys():
    kwargs = _base_optimizer_kwargs()
    kwargs["w_physics"] = 2.5
    kwargs["w_bc"] = 3.5

    optimizer = TrajectoryOptimizer(_build_model(), **kwargs)

    assert optimizer.config.physics_loss_weight == pytest.approx(2.5)
    assert optimizer.config.boundary_loss_weight == pytest.approx(3.5)


def test_optimizer_canonical_key_wins_over_alias():
    kwargs = _base_optimizer_kwargs()
    kwargs["w_physics"] = 2.5
    kwargs["physics_loss_weight"] = 4.0

    optimizer = TrajectoryOptimizer(_build_model(), **kwargs)

    assert optimizer.config.physics_loss_weight == pytest.approx(4.0)


def test_optimizer_does_not_train_in_constructor():
    optimizer = TrajectoryOptimizer(_build_model(), **_base_optimizer_kwargs())
    assert optimizer.last_run is None


def test_optimizer_accepts_optional_thrust_cap_settings():
    kwargs = _base_optimizer_kwargs()
    kwargs["thrust_cap"] = 0.1
    kwargs["thrust_cap_weight"] = 25.0

    optimizer = TrajectoryOptimizer(_build_model(), **kwargs)

    assert optimizer.config.thrust_cap == pytest.approx(0.1)
    assert optimizer.config.thrust_cap_weight == pytest.approx(25.0)


def test_optimizer_accepts_optional_tangential_thrust_smoothness_weight():
    kwargs = _base_optimizer_kwargs()
    kwargs["tangential_thrust_smoothness_weight"] = 3.5

    optimizer = TrajectoryOptimizer(_build_model(), **kwargs)

    assert optimizer.config.tangential_thrust_smoothness_weight == pytest.approx(3.5)


def test_optimizer_prefers_model_t_total_over_config_t_total():
    model = _build_model()
    model.register_parameter("t_total", torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32)))

    optimizer = TrajectoryOptimizer(model, **_base_optimizer_kwargs())
    run = optimizer.fit()

    assert float(run.t_total.detach().cpu().item()) != pytest.approx(1.0)
    assert float(run.t_total.detach().cpu().item()) == pytest.approx(
        float(model.t_total.detach().cpu().item())
    )


def test_execute_single_experiment_registers_non_optimizer_extra_parameters_only_on_model():
    config = {
        "label": "extra-param-check",
        "pinn": {
            "N_INPUT": 1,
            "N_OUTPUT": 2,
            "N_NEURONS": 8,
            "N_LAYERS": 2,
            "input_transform_fn": None,
            "output_transform_fn": None,
        },
        "extra_parameters": {
            "t_total": torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float32)),
            "alpha_N": torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float32)),
        },
        "optimizer": {
            **_base_optimizer_kwargs(),
            "show_progress": False,
        },
    }

    model, _result = execute_single_experiment(config)

    assert hasattr(model, "alpha_N")
    assert config["optimizer"]["t_total"] is model.t_total
    assert "alpha_N" not in config["optimizer"]
