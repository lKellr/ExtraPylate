import numpy as np
from numpy.typing import NDArray
from modules.helpers import *
from modules.root_finding import *
import pytest


def test_numerical_jacobian():
    def f(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [1.0 + x[0] * x[0] * x[1] - 4 * x[0], 3 * x[0] - x[0] * x[0] * x[1]]
        )

    def df(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [2 * x[0] * x[1] - 4.0, x[0] * x[0]],
                [3.0 - 2 * x[0] * x[1], -x[0] * x[0]],
            ]
        )

    x0 = np.array([2.0, 2.5])

    df_num = numerical_jacobian(x0, f, delta=1e-6)
    assert df_num == pytest.approx(df(x0)), (
        f"numerical jacobian df_num = {df_num} does not equal analytic value df = {df(x0)}"
    )


def test_numerical_jacobian_t():
    def f(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [t + x[0] * x[0] * x[1] - 4 * x[0], t * x[0] - x[0] * x[0] * x[1]]
        )

    def df(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [2 * x[0] * x[1] - 4.0, x[0] * x[0]],
                [t - 2 * x[0] * x[1], -x[0] * x[0]],
            ]
        )

    x0 = np.array([2.0, 2.5])
    t0 = 1.5

    df_num = numerical_jacobian_t(t0, x0, f, delta=1e-6)
    assert df_num == pytest.approx(df(t0, x0)), (
        f"numerical jacobian df_num = {df_num} does not equal analytic value df = {df(x0)}"
    )


@pytest.mark.parametrize("method", [Secant_method, Bisection])
def test_root_finding_scalar(method):
    def f(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return x * np.exp(-np.abs(x))  # x_0 = 0.0

    # f = lambda x: (x - 0.7) ** 4  # x_0 = 0.7
    # f = lambda x: 2 * (x - 0.7) + 0.03 * (x - 0.7) ** 3  # x_0 = 0.7
    # f = lambda x: np.clip(x, -1, 1)  # x_0 = 0.0
    bounds = [-3.0, 2.0]
    res = method(f, bounds[0], bounds[1], tol_iter=1e-5)
    assert res == pytest.approx(0.0, abs=1e-5, rel=1e-4)


@pytest.mark.parametrize(
    "method, additional_kwargs",
    [(Newton, {}), (root_wrapped, {}), (NewtonODE, {"eta_old": 1.0})],
)  # NewtonODE, root_wrapped
def test_root_finding_multivar(method, additional_kwargs):
    def f(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]
        )

    def jac(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array(
            [
                [1 + 1.5 * (x[0] - x[1]) ** 2, -1.5 * (x[0] - x[1]) ** 2],
                [-1.5 * (x[1] - x[0]) ** 2, 1 + 1.5 * (x[1] - x[0]) ** 2],
            ]
        )

    x0 = np.array([0.5, 0.0])
    x_sol = np.array([0.8411639, 0.1588361])
    res, success, info = method(f, x0, jac, tol_iter=1e-5, **additional_kwargs)
    assert success and res == pytest.approx(
        x_sol,
        abs=1e-5,
        rel=1e-4,
    ), f"error: {res - x_sol}\n{info}"
