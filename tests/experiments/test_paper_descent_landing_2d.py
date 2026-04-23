import math

from spacepinn.paper.descent_landing_2d import build_baseline_entry, build_config


def test_build_config_fixed_final_angle_uses_fixed_transform():
    config = build_config(alpha_final_pi=0.5, fixed_final_angle=True, smoke=True)

    assert "alpha_N" not in config["extra_parameters"]
    assert config["scenario"]["fixed_final_angle"] is True
    assert math.isclose(config["scenario"]["alpha_final_rad"], 0.5 * math.pi, rel_tol=0.0, abs_tol=1e-12)
    rN = config["optimizer"]["rN"].tolist()
    assert math.isclose(rN[0], 6378.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rN[1], 0.5 * math.pi, rel_tol=0.0, abs_tol=1e-6)


def test_build_baseline_entry_fixed_final_angle_records_constraint():
    entry = build_baseline_entry(alpha_final_pi=0.5, fixed_final_angle=True, baseline_max_iteration=100, smoke=True)
    config = entry["config"]

    assert config["solver"]["max_iteration"] == 1
    assert config["solver"]["num_states"] == [4]


def test_build_baseline_entry_atmosphere_records_drag_model():
    entry = build_baseline_entry(
        alpha_final_pi=0.5,
        fixed_final_angle=True,
        atmosphere=True,
        baseline_max_iteration=100,
        smoke=True,
    )
    config = entry["config"]

    assert config["spaceship"]["drag_model"] == "exponential"
    assert config["constraints"]["free_final_angle"] is False
    assert config["constraints"]["free_final_angle"] is False
    assert math.isclose(config["constraints"]["fixed_terminal_angle"], 0.5 * math.pi, rel_tol=0.0, abs_tol=1e-12)
