from functools import partial
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
import pandas as pd
from scipy.integrate import solve_ivp

from modules.helpers import norm_hairer
from modules.post_processing import ODEProblem
from modules.step_control import (
    StepControllerExtrapBulirsch,
    StepControllerExtrapK,
    StepControllerExtrapH,
    StepControllerExtrapKH_HW,
    StepControllerExtrapKH_Deuflhard,
)
from solvers.embedded import *
from solvers.explicit import *
from solvers.Extrapolation_Scheme import *
from time import perf_counter
import logging

logging.basicConfig(level=logging.INFO)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_pil = logging.getLogger("PIL")
logger_pil.setLevel(logging.INFO)

cmap = plt.get_cmap("tab20")


def get_run_extrap(extrap_scheme):
    def run_extrap(ode_fun, x0, t_max, atol, rtol):
        solver_extrap = extrap_scheme(
            ode_fun,
            table_size=8,
            step_controller=StepControllerExtrapKH_HW(atol=atol, rtol=rtol),
        )
        return solver_extrap.solve(x0, t_max)

    return run_extrap


def get_run_scipy(scheme_name):
    def run_sp(ode_fun, x0, t_max, atol, rtol):
        sol = solve_ivp(
            fun=ode_fun,
            t_span=(0.0, t_max),
            y0=x0,
            method=scheme_name,
            atol=atol,
            rtol=rtol,
        )
        return (
            sol.t,
            sol.y.T,
            dict(
                n_feval=sol.nfev,
                n_jaceval=sol.njev,
                n_lu=sol.nlu,
                n_restarts=0,
                local_errors=[],
            ),
        )

    return run_sp


def run_problem(ode_problem: ODEProblem, solver_runfunc, solver_run_kwargs=dict()):
    t_start = perf_counter()
    time, solution, solve_info = solver_runfunc(
        ode_fun=ode_problem.x_dot,
        x0=ode_problem.x0,
        t0=ode_problem.t_range[0],
        t_max=ode_problem.t_range[1],
        **solver_run_kwargs,
    )
    time_elapsed = perf_counter() - t_start

    error = norm(
        solution - ode_problem.x_analytic(time)
    )  # error over whole integration

    return error, time_elapsed, solve_info["n_feval"]


def benchmark_solver_controlled(
    ode_problem: ODEProblem, solver_runfunc, precision_list
):
    run_results_list = []

    for prec in precision_list:
        solver_run_kwargs = dict(atol=prec, rtol=prec)
        run_results = run_problem(ode_problem, solver_runfunc, solver_run_kwargs)

        run_results_list.append(run_results)

        if (
            run_results[1] > 1e1 or run_results[0] > 1.0
        ):  # finish for divergence or long times
            break
    dat_solver = pd.DataFrame(
        run_results_list, columns=["errors", "timings", "f_evals"]
    )
    return dat_solver


def benchmark_solver_stepsize(
    ode_problem: ODEProblem, solver_runfunc, precision_list, p_expected, h_initial=0.1
):
    h_last = h_initial  # initial h
    error_last = precision_list[0]  # disable reduction by p for first estimate

    run_results_list = []
    for prec in precision_list:
        h_prec = h_last * (prec / error_last) ** (1 / p_expected)
        solver_run_kwargs = dict(h=h_prec)

        run_results = run_problem(ode_problem, solver_runfunc, solver_run_kwargs)
        run_results_list.append(run_results)

        h_last = h_prec
        error_last = run_results[0]

        if (
            run_results[1] > 1e1 or error_last > 1.0
        ):  # finish for divergence or long times
            break

    dat_solver = pd.DataFrame(
        run_results_list, columns=["errors", "timings", "f_evals"]
    )
    return dat_solver

# benchmark problem / log problem
def x_dot_logproblem(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.array(
        [
            2 * t * x[0] * np.log(np.maximum(x[1], 1e-3)),
            -2 * t * x[1] * np.log(np.maximum(x[0], 1e-3)),
        ]
    )


def x_analytic_logproblem(t: float) -> NDArray[np.floating]:
    return np.array([np.exp(np.sin(t * t)), np.exp(np.cos(t * t))]).T

## Duffing oscillator
alpha = -1.0
beta = 1.0
gamma = 3.0
delta = 0.02
omega = 1.0


def x_dot_Duffing(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.array(
        [
            x[1],
            gamma * np.cos(omega * t)
            - (delta * x[1] + alpha * x[0] + beta * x[0] ** 3),
        ]
    )


# ode_problem = ODEProblem(
#     x_dot=x_dot_logproblem,
#     t_range=(0,5.0),
#     x0=np.array([1.0, np.e]),
#     x_analytic=x_analytic_logproblem,
# )
ode_problem = ODEProblem.with_numerical_reference_solution(
    x_dot=x_dot_Duffing,
    t_range=(0, 8 * np.pi),
    x0=np.array([1.0, 0]),
)

precision_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
solver_list = [DP54]
norm = norm_hairer

data = dict()

data["Euler"] = benchmark_solver_stepsize(
    ode_problem, Euler, precision_list[:3], p_expected=1.0, h_initial=1e-2
)
data["Heun"] = benchmark_solver_stepsize(
    ode_problem, Heun, precision_list[:4], p_expected=2.0, h_initial=1e-1
)
data["AB_5"] = benchmark_solver_stepsize(
    ode_problem,
    partial(AB_k, k=5),
    precision_list[1:],
    p_expected=5.0,
    h_initial=1e-2,  # this diverges for low precision / too large step sizes
)
data["SSPRK3"] = benchmark_solver_stepsize(
    ode_problem, SSPRK3, precision_list[:4], p_expected=3.0, h_initial=1e-1
)
data["DP54"] = benchmark_solver_controlled(ode_problem, DP54, precision_list)
data["RKX4"] = benchmark_solver_controlled(ode_problem, RKX4, precision_list)
data["EULEX"] = benchmark_solver_controlled(
    ode_problem, get_run_extrap(EulerExtrapolation), precision_list
)
data["ODEX"] = benchmark_solver_controlled(
    ode_problem, get_run_extrap(ModMidpointExtrapolation), precision_list
)
data["SP_RK45"] = benchmark_solver_controlled(
    ode_problem, get_run_scipy("RK45"), precision_list
)
data["SP_DOP853"] = benchmark_solver_controlled(
    ode_problem, get_run_scipy("DOP853"), precision_list
)

#  efficiency
fig, ax = plt.subplots()

ax.set_title("Work-Precision")
ax.set_xlabel("function evaluations")
ax.set_ylabel("error")
ax.set_yscale("log")
ax.set_xlim(0, 2e3)

for i, (scheme_name, dat_solver) in enumerate(data.items()):
    ax.plot(
        dat_solver["f_evals"],
        dat_solver["errors"],
        label=scheme_name,
        marker="o",
        color=cmap(i),
    )
ax.legend(frameon=False)
# fig.savefig("work_precision.png")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_title("Time-Precision")
ax.set_xlabel("time")
ax.set_ylabel("error")
ax.set_yscale("log")
ax.set_xlim(0, 1.0)

for i, (scheme_name, dat_solver) in enumerate(data.items()):
    ax.plot(
        dat_solver["timings"],
        dat_solver["errors"],
        label=scheme_name,
        marker="o",
        color=cmap(i),
    )
ax.legend(frameon=False)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
ax.set_title("Work-Time")
ax.set_xlabel("f_evals")
ax.set_ylabel("time")

for i, (scheme_name, dat_solver) in enumerate(data.items()):
    ax.plot(
        dat_solver["f_evals"],
        dat_solver["timings"],
        label=scheme_name,
        marker="o",
        color=cmap(i),
    )
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
