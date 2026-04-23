# %%
import numpy as np

from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics

from spacepinn.result import TrajectoryResult
from spacepinn.config.config_goddard import config_goddard
from spacepinn.opengoddard.legendre_patch import patch_opengoddard_legendre
from spacepinn.opengoddard._solve import solve_with_diagnostics


class Spaceship:
    def __init__(self):
        self.m = 1.0
        self.x0 = -1.0
        self.y0 = -1.0
        self.xf = 1.0
        self.yf = 1.0
        self.u_max = 0.0
        self.ao = np.array([[-0.5, -1.0, 0.5], [-0.2, 0.4, 1.0], [0.8, 0.3, 0.5]])  # astronomical objects
        self.r0 = np.array([self.x0, self.y0])
        self.rf = np.array([self.xf, self.yf])

    def compute_gravity_cartesian(self, x, y):
        """
        x, y: (N,) arrays of particle positions
        self.ao_rgm: (M, 3) array -> [x_ao, y_ao, gravitational_constant]
        self.G: (N, 2) array (will be updated in place)
        """

        # Construct r from x and y
        r = np.stack((x, y), axis=1)  # shape (N, 2)
        G = np.zeros_like(r)

        for ao in self.ao:
            ao_pos = ao[:2]  # (2,)
            G_const = ao[2]

            r_diff = r - ao_pos  # (N, 2)
            distances = np.linalg.norm(r_diff, axis=1) + 1e-15  # (N,)
            denominator = distances**3  # (N,)

            # Broadcast division over x and y components
            G -= G_const * r_diff / denominator[:, None]
        return G


def geometric_2d_opengoddard(
    label="Direct collocation",
    *,
    ftol: float = 1e-12,
    max_iteration: int = 5,
    slsqp_maxiter: int = 25,
):
    patch_opengoddard_legendre(Problem)

    def dynamics(prob: Problem, obj: Spaceship, section):
        # State variables: 2D Postion and Velocity
        x = prob.states(0, section)
        y = prob.states(1, section)
        vx = prob.states(2, section)
        vy = prob.states(3, section)

        # Control variables: 2D Thrust
        ux = prob.controls(0, section)
        uy = prob.controls(1, section)

        G = obj.compute_gravity_cartesian(x, y)
        dx = Dynamics(prob, section)
        dx[0] = vx
        dx[1] = vy
        dx[2] = ux / obj.m + G[:, 0]
        dx[3] = uy / obj.m + G[:, 1]

        return dx()

    def equality(prob: Problem, obj: Spaceship):
        x = prob.states_all_section(0)
        y = prob.states_all_section(1)

        result = Condition()

        # Initial / final conditions
        result.equal(x[0], obj.x0)
        result.equal(y[0], obj.y0)
        result.equal(x[-1], obj.xf)
        result.equal(y[-1], obj.yf)

        return result()

    def inequality(prob: Problem, obj: Spaceship):
        tf = prob.time_final(-1)

        result = Condition()
        result.lower_bound(tf, 0.0)

        return result()

    # Mayer Cost
    def cost(prob: Problem, obj: Spaceship):
        return 0.0

    # Lagrange Cost
    def running_cost(prob: Problem, obj: Spaceship):
        ux = prob.controls_all_section(0)
        uy = prob.controls_all_section(1)

        return ux**2 + uy**2

    # ---------------------------------------------

    # Initial TOF Guess
    time_init = [0.0, 1.0]
    n = [100]  # Collocation Points
    num_states = [4]  # Number of state variables
    num_controls = [2]  # Number of control variables
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Spaceship()

    opengoddard_config = {
        "backend": "OpenGoddard",
        "problem": "geometric_2d",
        "solver": {
            "time_init": time_init,
            "n": n,
            "num_states": num_states,
            "num_controls": num_controls,
            "max_iteration": max_iteration,
            "slsqp_maxiter": slsqp_maxiter,
            "ftol": ftol,
        },
        "spaceship": {
            "mass": obj.m,
            "r0": obj.r0,
            "rN": obj.rf,
            "gravity_sources": obj.ao,
        },
        "initial_guess": {
            "x": {"kind": "linear", "start": obj.x0, "end": obj.xf},
            "y": {"kind": "linear", "start": obj.y0, "end": obj.yf},
            "vx": {"kind": "constant", "value": 1.0},
            "vy": {"kind": "constant", "value": 1.0},
            "ux": {"kind": "constant", "value": 0.0},
            "uy": {"kind": "constant", "value": 0.0},
        },
    }

    # Initial Guess for Trajectory: Straight line from r0 -> rf
    x_init = Guess.linear(prob.time_all_section, obj.x0, obj.xf)
    y_init = Guess.linear(prob.time_all_section, obj.y0, obj.yf)

    # Initial Guess for Velocity: Constant mag sqrt(2)
    vx_init = Guess.constant(prob.time_all_section, 1.0)
    vy_init = Guess.constant(prob.time_all_section, 1.0)

    # Initial Guess Thrust: Constant 0
    ux_init = Guess.constant(prob.time_all_section, 0.0)
    uy_init = Guess.constant(prob.time_all_section, 0.0)

    # Set initial guesses
    prob.set_states_all_section(0, x_init)
    prob.set_states_all_section(1, y_init)
    prob.set_states_all_section(2, vx_init)
    prob.set_states_all_section(3, vy_init)

    prob.set_controls_all_section(0, ux_init)
    prob.set_controls_all_section(1, uy_init)

    # ---------------------------------------------
    prob.dynamics = [dynamics]
    prob.cost = cost
    prob.running_cost = running_cost
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        tf = prob.time_final(-1)
        print("tf: {0:.5f}".format(tf))

    runtime_seconds, solver_metadata = solve_with_diagnostics(
        prob,
        obj,
        display_func,
        ftol=ftol,
        maxiter=slsqp_maxiter,
        label=label,
    )

    # ========================
    # Post Process
    # ------------------------
    # Convert parameter vector to variable
    t = prob.time_update()
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    vx = prob.states_all_section(2)
    vy = prob.states_all_section(3)
    ux = prob.controls_all_section(0)
    uy = prob.controls_all_section(1)
    Gxy = obj.compute_gravity_cartesian(x, y)
    running_cost_values = np.asarray(running_cost(prob, obj), dtype=float)
    total_cost = float(cost(prob, obj) + np.trapezoid(running_cost_values, t))

    r = np.stack([x, y], axis=1)
    v = np.stack([vx, vy], axis=1)
    F = np.stack([ux, uy], axis=1)

    result = TrajectoryResult.from_open_goddard(
        label,
        t,
        r,
        v,
        F,
        Gxy,
        obj.r0,
        obj.rf,
        obj.ao,
        "cartesian",
        total_cost=total_cost,
        runtime_seconds=runtime_seconds,
        solver=solver_metadata,
    )

    return {
        "label": result.label,
        "result": result,
        **config_goddard["plotting"],
        "model": None,
        "config": opengoddard_config,
    }


if __name__ == "__main__":
    from spacepinn.plotter import TrajectoryPlotter

    result = geometric_2d_opengoddard()
    p = TrajectoryPlotter([result])
    p.plot_traj_2d(plot_quiver=False)
