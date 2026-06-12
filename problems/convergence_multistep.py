import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from modules.step_control import StepControllerExtrapDummy
from solvers.explicit import *
from solvers.implicit import *
from solvers.Extrapolation_Scheme import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_pil = logging.getLogger("PIL")
logger_pil.setLevel(logging.INFO)

cmap = plt.get_cmap("tab20")


def x_dot(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    return x * (2.0 - np.sin(t))


def jac(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    return 2.0 - np.sin(t)


t_max = 1.0
x0 = np.array([2.0])


def x_analytic(t: float) -> NDArray[np.floating]:
    return 2 * np.exp(2 * t + np.cos(t) - 1.0)


norm = norm_hairer

N_list = 2 * np.array(
    [
        2 ** (k // 2) if k == 1 or k % 2 == 0 else 1.5 * 2 ** (k // 2)
        for k in range(1, 16)
    ]
)
conv_data = dict()

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = BDF2(x_dot, x0, t_max, h=t_max / n_steps)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["BDF2"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = BDF3(x_dot, x0, t_max, h=t_max / n_steps)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["BDF3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 2 * t_max / n_steps, 3), 1)
    x_start = x_analytic(t_start)

    time, result, solve_info = BDF3(
        x_dot, x0, t_max, h=t_max / n_steps, x_start=x_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["BDF3started"] = np.array(errors), np.array(h_mins)


errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AM_k(x_dot, x0, t_max, h=t_max / n_steps, k=3)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AMk3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AM_k(x_dot, x0, t_max, h=t_max / n_steps, k=5)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AMk5"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 3 * t_max / n_steps, 4), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])

    time, result, solve_info = AM_k(
        x_dot, x0, t_max, h=t_max / n_steps, k=5, x_start=x_start, f_start=f_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AMk5started"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 4 * t_max / n_steps, 5), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])

    time, result, solve_info = AM_k(
        x_dot,
        x0,
        t_max,
        h=t_max / n_steps,
        k=6,
        x_start=x_start,
        f_start=f_start,
        solvertol=1e-12,
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AMk6started"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 5 * t_max / n_steps, 6), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])

    time, result, solve_info = AM_k(
        x_dot,
        x0,
        t_max,
        h=t_max / n_steps,
        k=7,
        x_start=x_start,
        f_start=f_start,
        solvertol=1e-12,
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AMk7_started"] = np.array(object=errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AB2(x_dot, x0, t_max, h=t_max / n_steps)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AB2"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, t_max / n_steps, 2), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])

    time, result, solve_info = AB2(
        x_dot, x0, t_max, h=t_max / n_steps, x_start=x_start, f_start=f_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AB2started"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AB3(x_dot, x0, t_max, h=t_max / n_steps)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AB3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 2 * t_max / n_steps, 3), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])

    time, result, solve_info = AB3(
        x_dot, x0, t_max, h=t_max / n_steps, x_start=x_start, f_start=f_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AB3_started"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AB_k(x_dot, x0, t_max, h=t_max / n_steps, k=3)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["ABk3"] = np.array(object=errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 2 * t_max / n_steps, 3), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])
    time, result, solve_info = AB_k(
        x_dot, x0, t_max, h=t_max / n_steps, k=3, x_start=x_start, f_start=f_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["ABk3_started"] = np.array(object=errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 5 * t_max / n_steps, 6), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])
    time, result, solve_info = AB_k(
        x_dot, x0, t_max, h=t_max / n_steps, k=6, x_start=x_start, f_start=f_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["ABk6_started"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AB_k(x_dot, x0, t_max, h=t_max / n_steps, k=7)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["ABk7"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    t_start = np.expand_dims(np.linspace(0.0, 6 * t_max / n_steps, 7), 1)
    x_start = x_analytic(t_start)
    f_start = x_dot(t_start[:-1], x_start[:-1])
    time, result, solve_info = AB_k(
        x_dot, x0, t_max, h=t_max / n_steps, k=7, x_start=x_start, f_start=f_start
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["ABk7_started"] = np.array(errors), np.array(h_mins)

# results
fig, ax = plt.subplots(figsize=(9.6, 7.2))
# ax.set_ylim(-5, 5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("H")
ax.set_ylabel("error")

secax = ax.secondary_xaxis(
    "top",
    functions=(lambda h: t_max / h, lambda n_steps: t_max / n_steps),
    xscale="linear",
)
secax.set_xlabel("steps")
# secax.tick_params(axis="x", which="minor")


for i, (scheme_name, (errors, _)) in enumerate(conv_data.items()):
    ax.plot(
        t_max / N_list,
        errors,
        label=scheme_name,
        color=cmap(i),
        marker="o",
        linestyle="--" if scheme_name in ["AB3", "RK4"] else "-",
    )
    rate = np.log(errors[1:] / errors[:-1]) / np.log(N_list[:-1] / N_list[1:])
    ax.text(
        t_max / N_list[len(N_list) // 2],
        errors[len(N_list) // 2],
        rf"$\bar{{p}} = {np.mean(rate):.2f}$",
    )
    ax.text(t_max / N_list[0], errors[0], f"$p_0 = {rate[0]:.2f}$")
    ax.text(t_max / N_list[-1], errors[-1], f"$p_\\infty = {rate[-1]:.2f}$")
    for k in range(len(N_list) - 1):
        ax.text(
            t_max / N_list[k],
            errors[k],
            f"$p_{k} = {rate[k]:.2f}$\n$N = {N_list[k]}\\to{N_list[k + 1]}$",
        )


plt.legend(frameon=False)
plt.tight_layout()
# plt.savefig("convergence.png")
plt.show()

# plot over minimum h
fig, ax = plt.subplots(figsize=(9.6, 7.2))
# ax.set_ylim(-5, 5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$h_\mathrm{min}$")
ax.set_ylabel("error")

for i, (scheme_name, (errors, h_min)) in enumerate(conv_data.items()):
    ax.plot(
        h_min,
        errors,
        label=scheme_name,
        color=cmap(i),
        marker="o",
    )
    rate = np.log(errors[1:] / errors[:-1]) / np.log(h_min[1:] / h_min[:-1])
    ax.text(
        h_min[len(h_min) // 2],
        errors[len(h_min) // 2],
        rf"$\bar{{p}} = {np.mean(rate):.2f}$",
    )
    ax.text(h_min[0], errors[0], f"$p_0 = {rate[0]:.2f}$")
    ax.text(h_min[-1], errors[-1], f"$p_\\infty = {rate[-1]:.2f}$")


plt.legend(frameon=False)
plt.tight_layout()
plt.show()
