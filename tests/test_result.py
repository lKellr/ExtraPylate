import numpy as np
from numpy.typing import NDArray
import pytest
from solvers.embedded import *
from solvers.explicit import *
from solvers.implicit import *
from solvers.Extrapolation_Scheme import *

np.set_printoptions(precision=20)


class TestResult:
    n_steps = 10

    @pytest.mark.parametrize(
        "scheme, expected_result, additional_kwargs",
        [
            (Euler, np.array([1.5176659044290977, 2.977254903226454]), {}),
            (Midpoint, np.array([1.5176844871672472, 2.9772274041165048]), {}),
            (Heun, np.array([1.5176844800849316, 2.9772274112298147]), {}),
            (AB2, np.array([1.5176844924745168, 2.9772274070953606]), {}),
            (AB3, np.array([1.5176844941534204, 2.977227393105639]), {}),
            (PECE, np.array([1.5176844938605085, 2.977227393398317]), {}),
            (PECE_tol, np.array([1.5176844938605085, 2.977227393398317]), {}),
            (PEC, np.array([1.5176844938609586, 2.9772273933977043]), {}),
            (RK4, np.array([1.5176844950225017, 2.97722739007441]), {}),
            (SSPRK3, np.array([1.5176844950191086, 2.977227390082689]), {}),
            (SSPRK34, np.array([1.5176844950208044, 2.977227390078549]), {}),
            (Backwards_Euler, np.array([1.517703139758524, 2.977199797748956]), {}),
            (BDF2, np.array([1.5176873241001914, 2.9772232101112146]), {}),
            (TRBDF2, np.array([1.5176844990087082, 2.9772273845315906]), {}),
            (BDF3, np.array([1.5176869063212122, 2.9772238341584223]), {}),
            (AB_k, np.array([1.517682622723628, 2.977230156477312]), {"k": 9}),
            (AM_k, np.array([1.5176844968530951, 2.9772273879339997]), {"k": 9}),
        ],
    )
    def test_scheme_Brusselator(
        self,
        scheme: Callable,
        expected_result: NDArray[np.floating],
        additional_kwargs: dict,
    ):
        def x_dot(t: float, x: NDArray[np.floating]):
            return np.array(
                [1.0 + x[0] * x[0] * x[1] - 4 * x[0], 3 * x[0] - x[0] * x[0] * x[1]],
                dtype=x.dtype,
            )

        x0 = np.array([1.5, 3.0])

        h = 1e-3
        t_max = self.n_steps * h
        time, result, solve_info = scheme(x_dot, x0, t_max, h, **additional_kwargs)

        assert (result[-1] == expected_result).all(), (
            f"Unexpected final result with x = {result[-1]}, should be {expected_result}, error: {result[-1] - expected_result}."
        )
