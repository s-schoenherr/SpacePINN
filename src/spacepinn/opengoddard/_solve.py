from __future__ import annotations

import time
from typing import Any

import numpy as np
from OpenGoddard.optimize import optimize


def _format_solver_detail(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_opengoddard_start(
    *,
    label: str | None,
    prob,
    ftol: float,
    maxiter: int,
    details: dict[str, Any] | None = None,
) -> None:
    run_label = label or "OpenGoddard"
    print()
    print("*" * 92)
    print(f"[SWINGBY] Starting experiment | label={run_label} | source=opengoddard")
    print(
        "[SWINGBY] OpenGoddard setup | "
        f"sections={getattr(prob, 'number_of_section', 'n/a')} | "
        f"states={getattr(prob, 'number_of_states', 'n/a')} | "
        f"outer_iterations={getattr(prob, 'maxIterator', 'n/a')} | "
        f"slsqp_maxiter={maxiter} | ftol={ftol:.3g}"
    )
    for key, value in (details or {}).items():
        print(f"[SWINGBY] {key}: {_format_solver_detail(value)}")
    print("*" * 92)
    print()


def solve_with_diagnostics(
    prob,
    obj,
    display_func,
    *,
    ftol: float,
    maxiter: int = 25,
    label: str | None = None,
    details: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run OpenGoddard's SLSQP loop and return process CPU time plus structured diagnostics."""
    assert len(prob.dynamics) != 0, "It must be set dynamics"
    assert prob.cost is not None, "It must be set cost function"
    assert prob.equality is not None, "It must be set equality function"
    assert prob.inequality is not None, "It must be set inequality function"
    print_opengoddard_start(label=label, prob=prob, ftol=ftol, maxiter=maxiter, details=details)

    def equality_add(equality_func, solve_obj):
        del equality_func
        result = prob.equality(prob, solve_obj)

        for section in range(prob.number_of_section):
            derivative = np.zeros(0)
            for state_index in range(prob.number_of_states[section]):
                state_temp = prob.states(state_index, section) / prob.unit_states[section][state_index]
                derivative = np.hstack((derivative, prob.D[section].dot(state_temp)))
            tix = prob.time_start(section) / prob.unit_time
            tfx = prob.time_final(section) / prob.unit_time
            dx = prob.dynamics[section](prob, solve_obj, section)
            result = np.hstack((result, derivative - (tfx - tix) / 2.0 * dx))

        for knot in range(prob.number_of_section - 1):
            if prob.number_of_states[knot] != prob.number_of_states[knot + 1]:
                continue
            for state_index in range(prob.number_of_states[knot]):
                param_prev = prob.states(state_index, knot) / prob.unit_states[knot][state_index]
                param_post = prob.states(state_index, knot + 1) / prob.unit_states[knot][state_index]
                if prob.knot_states_smooth[knot]:
                    result = np.hstack((result, param_prev[-1] - param_post[0]))

        return result

    def cost_add(cost_func, solve_obj):
        del cost_func
        not_integrated = prob.cost(prob, solve_obj)
        if prob.running_cost is None:
            return not_integrated
        integrand = prob.running_cost(prob, solve_obj)
        weight = np.concatenate([weights for weights in prob.w])
        integrated = sum(integrand * weight)
        return not_integrated + integrated

    def wrap_for_solver(func, arg0, arg1):
        def for_solver(p, arg0, arg1):
            prob.p = p
            return func(arg0, arg1)

        return for_solver

    cons = (
        {
            "type": "eq",
            "fun": wrap_for_solver(equality_add, prob.equality, obj),
            "args": (prob, obj),
        },
        {
            "type": "ineq",
            "fun": wrap_for_solver(prob.inequality, prob, obj),
            "args": (prob, obj),
        },
    )

    jac = None if prob.cost_derivative is None else wrap_for_solver(prob.cost_derivative, prob, obj)

    started = time.process_time()
    last_opt = None
    outer_iterations_completed = 0

    while prob.iterator < prob.maxIterator:
        print(f"---- iteration : {prob.iterator + 1} ----")
        opt = optimize.minimize(
            wrap_for_solver(cost_add, prob.cost, obj),
            prob.p,
            args=(prob, obj),
            constraints=cons,
            jac=jac,
            method="SLSQP",
            options={"disp": True, "maxiter": maxiter, "ftol": ftol},
        )
        last_opt = opt
        outer_iterations_completed = prob.iterator + 1
        print(opt.message)
        display_func()
        print("")
        if not opt.status:
            break
        prob.iterator += 1

    runtime_seconds = time.process_time() - started
    status_code = None if last_opt is None else int(last_opt.status)
    converged = status_code == 0 if status_code is not None else None

    if converged:
        stopped_reason = "converged"
    elif outer_iterations_completed >= int(prob.maxIterator):
        stopped_reason = "iteration_budget"
    elif status_code is None:
        stopped_reason = "not_run"
    else:
        stopped_reason = "solver_status"

    diagnostics = {
        "backend": "OpenGoddard",
        "converged": converged,
        "status_code": status_code,
        "message": None if last_opt is None else str(last_opt.message),
        "stopped_reason": stopped_reason,
        "outer_iterations_completed": int(outer_iterations_completed),
        "outer_iteration_budget": int(prob.maxIterator),
        "slsqp_maxiter": int(maxiter),
        "ftol": float(ftol),
        "objective_value": None if last_opt is None else float(last_opt.fun),
    }
    return runtime_seconds, diagnostics
