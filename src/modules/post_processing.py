from typing import Callable, NamedTuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from solvers.embedded import DP54
from modules.helpers import norm_hairer

class ODEProblem(NamedTuple):
    x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]]
    t_range: tuple[int, int]
    x0: NDArray[np.floating]
    x_analytic: Callable[[float], NDArray[np.floating]] | None = None

    @classmethod
    def with_numerical_reference_solution(
        cls,
        x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        t_range: tuple[int, int],
        x0: NDArray[np.floating],
        atol: float = 1e-9,
        rtol: float = 1e-6,
        scheme_name: str = "DOP853",
    ):
        cls_no_ref = cls(
            x_dot,
            t_range,
            x0,
        )
        return cls(
            x_dot,
            t_range,
            x0,
            create_reference_solution(
                cls_no_ref,
                atol,
                rtol,
                scheme_name,
            ),
        )


def find_local_errors(
    x_dot: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    t: NDArray[np.floating],
    x_computed: NDArray[np.floating],
    norm: Callable[[NDArray[np.floating]], np.floating] = norm_hairer,
) -> NDArray[np.floating]:
    """computes the local error using DP54"""
    err_loc = np.empty(x_computed.shape[0])
    err_loc[0] = 0.0
    for ix_time in range(t.size - 1):
        _, x_analytic, _ = DP54(
            x_dot,
            x0=x_computed[ix_time],
            t0=t[ix_time],
            t_max=t[ix_time + 1],
            h_limits=(1e-20, np.inf),
            atol=1e-16,
            rtol=1e-9,
        )
        err_loc[ix_time + 1] = norm(x_computed[ix_time + 1] - x_analytic[-1])
    return err_loc

def create_reference_solution(
    ode_problem: ODEProblem,
    atol: float = 1e-9,
    rtol: float = 1e-6,
    scheme_name: str = "DOP853",
) -> Callable[[float], NDArray[np.floating]]:
    solve_result = solve_ivp(
        ode_problem.x_dot,
        ode_problem.t_range,
        ode_problem.x0,
        scheme_name,
        atol=atol,
        rtol=rtol,
        dense_output=True,
    )

    if not solve_result.success:
        print(f"Creation of reference solution failed tue to {solve_result.message}")

    # t_high = sol.t
    # x_high = sol.y.T

    def x_ref(t: float) -> Callable[[float], NDArray[np.floating]]:
        # return np.array(
        #     [np.interp(t, t_high, x_high[:, i]) for i in range(ode_problem.x0.size)]
        # ).T
        return solve_result.sol(t).T  # transpose result from dense interpolant

    return x_ref