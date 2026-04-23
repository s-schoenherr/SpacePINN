from __future__ import annotations

import importlib
import math


def test_build_scenario_matches_requested_eci_boundary_values():
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")
    shared = importlib.import_module(
        "spacepinn.paper._rendezvous_hold_point_eci_shared"
    )

    scenario = module.build_scenario(t_final_seconds=900.0)
    target_speed = math.sqrt(shared.GM_EARTH / (6878.0**3)) * 6878.0

    assert scenario["t_final_seconds"] == 900.0
    assert scenario["target"]["radius_km"] == 6878.0
    assert math.isclose(float(scenario["target"]["speed_km_s"]), target_speed, rel_tol=0.0, abs_tol=1e-12)

    start_target = scenario["target"]["start"]["position"]
    start_velocity = scenario["target"]["start"]["velocity"]
    start_chaser = scenario["chaser"]["start_position_km"]
    start_chaser_velocity = scenario["chaser"]["start_velocity_km_s"]

    assert math.isclose(float(start_target[0]), 6878.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_target[1]), 0.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_velocity[0]), 0.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_velocity[1]), target_speed, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(start_chaser[0]), 6877.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_chaser[1]), -0.3, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_chaser_velocity[0]), 0.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_chaser_velocity[1]), target_speed, rel_tol=0.0, abs_tol=1e-12)
    start_chaser_polar = scenario["chaser"]["start_position_polar"]
    start_velocity_polar = scenario["chaser"]["start_velocity_polar"]
    assert math.isclose(float(start_chaser_polar[0]), math.hypot(6877.0, -0.3), rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(start_chaser_polar[1]), math.atan2(-0.3, 6877.0), rel_tol=0.0, abs_tol=1e-12)
    expected_vr = ((6877.0 * 0.0) + (-0.3 * target_speed)) / math.hypot(6877.0, -0.3)
    expected_vt = ((-(-0.3) * 0.0) + (6877.0 * target_speed)) / math.hypot(6877.0, -0.3)
    assert math.isclose(float(start_velocity_polar[0]), expected_vr, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(start_velocity_polar[1]), expected_vt, rel_tol=0.0, abs_tol=1e-12)

    target_end = scenario["target"]["end"]["position"]
    hold_point = scenario["chaser"]["end_position_km"]
    radial = scenario["target"]["end"]["radial_unit"]
    expected_hold_point = target_end + 0.03 * radial

    assert math.isclose(float(hold_point[0]), float(expected_hold_point[0]), rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(hold_point[1]), float(expected_hold_point[1]), rel_tol=0.0, abs_tol=1e-9)
    hold_point_speed = (6878.0 + 0.03) * scenario["target"]["end"]["mean_motion"]
    assert math.isclose(
        float(scenario["chaser"]["end_velocity_km_s"][0]),
        float(hold_point_speed * scenario["target"]["end"]["along_track_unit"][0]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(scenario["chaser"]["end_velocity_km_s"][1]),
        float(hold_point_speed * scenario["target"]["end"]["along_track_unit"][1]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_build_config_sets_polar_exact_bc_setup():
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")

    config = module.build_config(t_final_seconds=1200.0, smoke=False)

    assert config["label"] == module.PINN_LABEL
    assert config["optimizer"]["coordinate_system"] == "polar"
    assert int(config["optimizer"]["n_adam"]) == module.PAPER_N_ADAM
    assert int(config["optimizer"]["n_lbfgs"]) == module.PAPER_N_LBFGS
    assert math.isclose(float(config["optimizer"]["convergence_threshold"]), module.PAPER_CONVERGENCE_THRESHOLD, rel_tol=0.0)
    assert math.isclose(float(config["optimizer"]["t_total"].detach().cpu().item()), 1200.0, rel_tol=0.0, abs_tol=1e-9)
    assert tuple(config["optimizer"]["r0"].shape) == (2,)
    assert tuple(config["optimizer"]["rN"].shape) == (2,)
    assert bool(config["extra_parameters"]["t_total"].requires_grad) is True

    smoke_config = module.build_config(smoke=True)
    assert int(smoke_config["optimizer"]["n_adam"]) == 1
    assert int(smoke_config["optimizer"]["n_lbfgs"]) == 0


def test_build_config_accepts_training_budget_overrides():
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")

    config = module.build_config(n_adam=42, n_lbfgs=7, convergence_threshold=1e-8, smoke=False)

    assert int(config["optimizer"]["n_adam"]) == 42
    assert int(config["optimizer"]["n_lbfgs"]) == 7
    assert math.isclose(float(config["optimizer"]["convergence_threshold"]), 1e-8, rel_tol=0.0)


def test_build_config_can_enable_target_angle_guard():
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")

    config = module.build_config(enforce_alpha_guard=True, smoke=False)

    assert config["label"] == module.PINN_GUARDED_LABEL
    assert config["plotting"]["color"] == module.PINN_GUARDED_COLOR
    transform_repr = repr(config["pinn"]["output_transform_fn"])
    assert "kinematic_rendezvous_hold_point_eci_polar_alpha_guard_fn" in transform_repr


def test_transform_terminal_state_depends_on_trainable_t_total():
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")

    config = module.build_config(t_final_seconds=900.0, smoke=False)
    transform_fn = config["pinn"]["output_transform_fn"]

    class _Model:
        def __init__(self, t_total):
            self.t_total = t_total

    tau = module.torch.tensor([[0.0], [1.0]], dtype=module.torch.float32)
    raw = module.torch.zeros((2, 2), dtype=module.torch.float32)
    nominal = transform_fn(tau, raw, model=_Model(module.torch.tensor(900.0)))
    later = transform_fn(tau, raw, model=_Model(module.torch.tensor(1200.0)))

    assert module.torch.allclose(nominal[0], later[0])
    assert not module.torch.allclose(nominal[-1], later[-1])
    assert math.isclose(float(nominal[-1, 0]), 6878.03, rel_tol=0.0, abs_tol=5e-4)


def test_sync_dynamic_terminal_reference_updates_result_boundary():
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")

    class _Dynamics:
        def __init__(self):
            self.rN = None

    class _Result:
        def __init__(self):
            self.t_total = 900.0
            self.rN = None
            self.dynamics = _Dynamics()

    result = _Result()
    scenario = module.build_scenario(t_final_seconds=900.0)

    module._sync_dynamic_terminal_reference(result, scenario=scenario)

    hold = result.dynamic_terminal_reference["hold_point_position_km"]
    assert result.rN.shape == (2,)
    assert module.np.allclose(result.rN, hold)
    assert module.np.allclose(result.dynamics.rN, hold)


def test_main_uses_collection_label(monkeypatch):
    module = importlib.import_module("spacepinn.paper.rendezvous_hold_point_eci")
    captured = {}

    def fake_run_collection(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "label": module.COLLECTION_LABEL,
            "entries": [],
            "plot_output_dir": str(module.RUN_ROOT),
            "run_dir": str(module.RUN_ROOT),
            "scenario": module.build_scenario(),
        }

    monkeypatch.setattr(
        module,
        "run_collection",
        fake_run_collection,
    )
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(module, "plot_results", lambda *args, **kwargs: None)

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["kwargs"]["t_final_seconds"] == module.DEFAULT_T_FINAL_SECONDS
