from functools import partial
import logging

from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from modules.helpers import (
    norm_hairer,
    numerical_jacobian_t,
)
from modules.root_finding import NewtonODE

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def Backwards_Euler(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = NewtonODE,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backwards Euler Method. System of Equations solved by solver(ode_fun==0, a, b, tol_iter)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        logger.warning(
            f"final step not hitting t_max exactly, instead t_max = {steps * h}"
        )

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    if jac_fun is None:
        jac_fun = lambda t, x: numerical_jacobian_t(
            t,
            x,
            ode_fun,
            delta=1e-8,
        )

    def f_imp_Newton(
        x_next: NDArray[np.floating], t_i: float, x_i: NDArray[np.floating]
    ):
        return x_next - x_i - h * ode_fun(t_i, x_next)

    if jac_fun is None:

        def jac_imp_Newton(x_next: NDArray[np.floating], t_i: float):
            return np.eye(x0.shape[0]) - h * numerical_jacobian_t(
                t_i,
                x_next,
                ode_fun,
                delta=1e-8,
            )

    else:

        def jac_imp_Newton(x_next: NDArray[np.floating], t_i: float):
            return np.eye(x0.shape[0]) - h * jac_fun(t_i, x_next)

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    x[0] = x0
    sol_info = dict(eta=np.inf)
    for i in range(steps):
        # f_imp = lambda x_next: x_next - x[i] - h * ode_fun(t[i + 1], x_next)
        # jac_imp = lambda x_next: np.eye(x0.shape[0]) - h * jac_fun(t[i + 1], x_next)
        x[i + 1], success, sol_info = nl_solver(
            fun=partial(f_imp_Newton, t_i=t[i + 1], x_i=x[i]),
            x0=x[i],
            tol_iter=norm_hairer(solvertol * x[i]) + solvertol,
            jac_fun=partial(jac_imp_Newton, t_i=t[i + 1]),
            norm=norm_hairer,
            eta_old=sol_info["eta"],
        )
        if not success:
            logger.warning(
                f"solver did not converge, reason: {sol_info['stop_reason']}"
            )
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]

    return t, x, info


def AM_k(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    k: int,
    t0: float = 0.0,
    x_start: NDArray[np.floating] | None = None,
    f_start: NDArray[np.floating] | None = None,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = NewtonODE,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams-Moulton formula of variable order k, maximum implemented is 9"""
    if x_start is not None:
        assert x_start.shape == (
            k - 1,
            x0.shape[0],
        ), (
            f"wrong shape of starting values x_start {x_start.shape}, should be {(k - 1, x0.shape[0])}"
        )
        assert x_start[0] == x0, "first value of x_start must equal x0"
    if f_start is not None:
        assert f_start.shape == (
            k - 1,
            x0.shape[0],
        ), (
            f"wrong shape of starting values f_start {f_start.shape}, should be {(k - 1, x0.shape[0])}"
        )

    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        logger.warning(
            f"final step not hitting t_max exactly, instead t_max = {steps * h}"
        )
    if steps + 2 < k:
        logger.warning(
            f"Number of steps {steps} not sufficient to reach target order {k}"
        )
        k = steps
        if x_start is not None:
            x_start = x_start[: k - 1]
        if f_start is not None:
            f_start = f_start[: k - 1]

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x, info, _ = _AM_k(
        ode_fun=ode_fun,
        x0=x0,
        steps=steps,
        h=h,
        k=k,
        t0=t0,
        x_start=x_start,
        f_start=f_start,
        nl_solver=nl_solver,
        jac_fun=jac_fun,
        solvertol=solvertol,
    )
    return t, x, info


def _AM_k(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    steps: int,
    h: float,
    k: int,
    t0: float = 0.0,
    x_start: NDArray[np.floating] | None = None,
    f_start: NDArray[np.floating] | None = None,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = NewtonODE,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], dict[str, Any], NDArray[np.floating]]:
    """Adams-Moulton of variable order k, this function also returns the computed function values"""
    assert k <= 9, "highest implemented order is 9"

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    def f_imp_Newton(
        x_next: NDArray[np.floating], t_i: float, f_const: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return x_next - (f_const + h * beta[0] * ode_fun(t_i, x_next))

    if jac_fun is None:

        def jac_imp_Newton(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x_next.shape[0]) - h * beta[0] * numerical_jacobian_t(
                t_i,
                x_next,
                ode_fun,
                delta=1e-8,
            )

    else:

        def jac_imp_Newton(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x_next.shape[0]) - h * beta[0] * jac_fun(t_i, x_next)

    # compute the coefficients
    gamma = [
        1.0,
        -1 / 2,
        -1 / 12,
        -1 / 24,
        -19 / 720,
        -3 / 160,
        -863 / 60480,
        -275 / 24192,
        -22953 / 3628800,
    ]
    beta: NDArray[Any] = np.array(
        [
            (-1) ** (j - 1)
            * sum([gamma[i] * comb(i, j - 1, exact=True) for i in range(j - 1, k)])
            for j in range(1, k + 1)
        ]
    )
    # NOTE: i am not sure about the (-1)**j term, it is not given in the Flaherty lecture notes, but results are wrong without it

    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    f_i = np.empty((k - 1, x0.shape[0]), dtype=x0.dtype)

    if k <= 1:  # start with the trapezoidal rule
        x[0] = x0
        f_i[0] = ode_fun(t0, x0)
    elif x_start is not None:
        x[: k - 1] = x_start
        if f_start is not None:
            f_i = f_start[::-1]
        else:
            for i in range(k - 1):
                f_i[k - 2 - i] = ode_fun(t0 + i * h, x_start[i])

    else:
        x[: k - 1], inf_starter, f_i = _AM_k(ode_fun, x0, k - 2, h, k - 1, t0)
        info = inf_starter

    steps_starter = k - 2 if k > 1 else 0
    sol_info = dict(eta=np.inf)
    for i in range(steps_starter, steps):
        f_i = np.roll(f_i, 1, axis=0)
        f_i[0] = ode_fun(t0, x[i])  # TODO: we overwrite one of the specified values

        f_const = (
            x[i] + h * beta @ f_i  # [1:] if k > 1 else x[i]
        )  # precompute the constant part
        x[i + 1], success, sol_info = nl_solver(
            fun=partial(f_imp_Newton, t_i=t0 + (i + 1) * h, f_const=f_const),
            x0=x[i],
            tol_iter=norm_hairer(solvertol * x[i]) + solvertol,
            jac_fun=partial(jac_imp_Newton, t_i=t0 + (i + 1) * h),
            norm=norm_hairer,
            eta_old=sol_info["eta"],
        )
        f_i[0] = ode_fun(t0 + (i + 1) * h, x[i + 1])

        if not success:
            logger.warning("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]
    return x, info, f_i


def BDF2(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    x_start: NDArray[np.floating] | None = None,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = NewtonODE,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backward differantiation Formula of order 2 for stiff systems.
    Starting values generated with backwards Euler method
    System of Equations solved by solver(ode_fun==0, a, b, tol_iter)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        logger.warning(
            f"final step not hitting t_max exactly, instead t_max = {steps * h}"
        )

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    def f_imp_Newton(
        x_next: NDArray[np.floating],
        t_i: float,
        x_i: NDArray[np.floating],
        x_ii: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        return x_next - 4 / 3 * x_i + 1 / 3 * x_ii - 2 / 3 * h * ode_fun(t_i, x_next)

    if jac_fun is None:

        def jac_imp_Newton(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x0.shape[0]) - 2 / 3 * h * numerical_jacobian_t(
                t_i,
                x_next,
                ode_fun,
                delta=1e-8,
            )

    else:

        def jac_imp_Newton(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x0.shape[0]) - 2 / 3 * h * jac_fun(
                t_i,
                x_next,
            )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    if x_start is not None:
        assert x_start.shape == (
            2,
            x0.shape[0],
        ), (
            f"wrong shape of starting values x_start {x_start.shape}, should be {(2, x0.shape[0])}"
        )
        assert x_start[0] == x0, "first value of x_start must equal x0"
        x[:2] = x_start
    else:
        t[:2], x[:2], inf_starter = Backwards_Euler(
            ode_fun=ode_fun,
            x0=x0,
            t_max=t0 + h,
            h=h,
            t0=t0,
            nl_solver=nl_solver,
            solvertol=solvertol,
        )
        info = inf_starter

    for i in range(1, steps):
        x[i + 1], success, sol_info = nl_solver(
            partial(f_imp_Newton, t_i=t[i + 1], x_i=x[i], x_ii=x[i - 1]),
            x0=x[i],
            tol_iter=norm_hairer(solvertol * x[i]) + solvertol,
            jac_fun=partial(jac_imp_Newton, t_i=t[i + 1]),
            norm=norm_hairer,
            eta_old=sol_info["eta"],
        )
        if not success:
            logger.warning("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]
    return t, x, info


def TRBDF2(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    solvertol: float = 1e-5,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = NewtonODE,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Combination of the trapezoidal method with BDF2 to get a DIRK scheme,
    see "Analysis and implementation of TR-BDF2", Hosea and Shampine 1996"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        logger.warning(
            f"final step not hitting t_max exactly, instead t_max = {steps * h}"
        )

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    def f_imp_Newton1(
        x_halftrapz: NDArray[np.floating], t_i: float, x_i: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return x_halftrapz - (
            x_i + 0.25 * h * (ode_fun(t_i, x_i) + ode_fun(t_i + 0.5 * h, x_halftrapz))
        )

    def f_imp_Newton2(
        x_next: NDArray[np.floating],
        t_i: float,
        x_i: NDArray[np.floating],
        x_halftrapz: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        return (
            x_next
            - 1.0
            / 3.0
            * (
                4 * x_halftrapz
                - x_i
                + h
                * ode_fun(
                    t_i + h, x_next
                )  # Note that the step is here half of what it is in the normal BDF2 scheme!
            )
        )

    if jac_fun is None:

        def jac_imp_Newton1(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x0.shape[0]) - numerical_jacobian_t(
                t_i + 0.5 * h,
                x_next,
                ode_fun,
                delta=1e-8,
            )

        def jac_imp_Newton2(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x0.shape[0]) - 1 / 3 * h * numerical_jacobian_t(
                t_i + h,
                x_next,
                ode_fun,
                delta=1e-8,
            )

    else:

        def jac_imp_Newton1(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x0.shape[0]) - jac_fun(t_i + 0.5 * h, x_next)

        def jac_imp_Newton2(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return np.eye(x0.shape[0]) - 1 / 3 * h * jac_fun(
                t_i + h,
                x_next,
            )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    x[0] = x0
    sol1_info = dict(eta=np.inf)
    sol2_info = dict(eta=np.inf)
    for i in range(steps):
        x_halftrapz, success, sol1_info = nl_solver(
            partial(f_imp_Newton1, t_i=t[i + 1], x_i=x[i]),
            x0=x[i],
            tol_iter=norm_hairer(solvertol * x[i]) + solvertol,
            jac_fun=partial(jac_imp_Newton1, t_i=t[i]),
            norm=norm_hairer,
            eta_old=sol1_info["eta"],
        )
        if not success:
            logger.warning("solver did not converge")
            break
        info["n_feval"] += sol1_info["n_feval"]
        info["n_jaceval"] += sol1_info["n_jaceval"]
        info["n_lu"] += sol1_info["n_lu"]

        x[i + 1], success, sol2_info = nl_solver(
            partial(f_imp_Newton2, t_i=t[i], x_i=x[i], x_halftrapz=x_halftrapz),
            x0=x_halftrapz,
            tol_iter=norm_hairer(solvertol * x[i]) + solvertol,
            jac_fun=partial(jac_imp_Newton2, t_i=t[i]),
            norm=norm_hairer,
            eta_old=sol2_info["eta"],
        )
        if not success:
            logger.warning("solver did not converge")
            break
        info["n_feval"] += sol2_info["n_feval"]
        info["n_jaceval"] += sol2_info["n_jaceval"]
        info["n_lu"] += sol2_info["n_lu"]
    return t, x, info


def BDF3(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    x_start: NDArray[np.floating] | None = None,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = NewtonODE,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backward differantiation Formula of order 3 for stiff systems.
    Starting values generated with backwards Euler method and BDF2
    System of Equations solved by solver(ode_fun==0, a, b, tol_iter)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        logger.warning(
            f"final step not hitting t_max exactly, instead t_max = {steps * h}"
        )

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    def f_imp_Newton(
        x_next: NDArray[np.floating],
        t_i: float,
        x_i: NDArray[np.floating],
        x_ii: NDArray[np.floating],
        x_iii: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        return (
            11 * x_next - 18 * x_i + 9 * x_ii - 2 * x_iii - 6 * h * ode_fun(t_i, x_next)
        )

    if jac_fun is None:

        def jac_imp_Newton(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return 11 * np.eye(x0.shape[0]) - 6 * h * numerical_jacobian_t(
                t_i,
                x_next,
                ode_fun,
                delta=1e-8,
            )

    else:

        def jac_imp_Newton(
            x_next: NDArray[np.floating], t_i: float
        ) -> NDArray[np.floating]:
            return 11 * np.eye(x0.shape[0]) - 6 * h * jac_fun(
                t_i,
                x_next,
            )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    if x_start is not None:
        assert x_start.shape == (
            3,
            x0.shape[0],
        ), (
            f"wrong shape of starting values x_start {x_start.shape}, should be {(3, x0.shape[0])}"
        )
        assert x_start[0] == x0, "first value of x_start must equal x0"
        x[:3] = x_start
    else:
        t[:3], x[:3], inf_starter = BDF2(
            ode_fun=ode_fun,
            x0=x0,
            t_max=t0 + 2 * h,
            h=h,
            t0=t0,
            nl_solver=nl_solver,
            solvertol=solvertol,
            jac_fun=jac_fun,
        )
        info = inf_starter

    for i in range(2, steps):
        x[i + 1], success, sol_info = nl_solver(
            partial(
                f_imp_Newton, t_i=t[i + 1], x_i=x[i], x_ii=x[i - 1], x_iii=x[i - 2]
            ),
            x0=x[i],
            tol_iter=norm_hairer(solvertol * x[i]) + solvertol,
            jac_fun=partial(jac_imp_Newton, t_i=t[i + 1]),
            norm=norm_hairer,
            eta_old=sol_info["eta"],
        )

        if not success:
            logger.warning("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]
    return t, x, info
