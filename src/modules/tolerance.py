from typing import Callable

import numpy as np
from numpy.typing import NDArray

from modules.helpers import clip, norm_hairer


class ToleranceManager:
    def __init__(
        self,
        atol: float | NDArray[np.floating],
        rtol: float | NDArray[np.floating],
        norm: Callable[[NDArray[np.floating]], np.floating] = norm_hairer,
    ):
        self.atol = atol
        self.rtol = rtol
        self.norm = norm

    def get_initial_stepFhty(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        x0: NDArray[np.floating],
        t_max: float,
        p: int,
        t0: float = 0.0,
    ) -> float:
        """From the Flaherty lecture notes"""
        tol = self.atol + self.rtol * np.abs(x0)

        step_size0 = (
            self.norm(tol) / (1.0 / (t_max - t0) ** p + self.norm(ode_fun(t0, x0)) ** p)
        ) ** (1 / p)
        return step_size0

    def get_initial_stepHW(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        x0: NDArray[np.floating],
        p: int,
        t0: float = 0.0,
    ) -> float:
        """From Hairer & Wanner eq. 4.14"""
        tol = self.atol + self.rtol * np.abs(x0)

        f0 = ode_fun(t0, x0)
        d0 = self.norm(x0 / tol)
        d1 = self.norm(f0 / tol)

        h0: float
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * (d0 / d1)

        x1_Eul = x0 + h0 * f0
        d2 = (
            self.norm((ode_fun(t0 + h0, x1_Eul) - f0) / tol) / h0
        )  # estimate of second derivative

        h1 = (0.01 / max(d1, d2)) ** (1.0 / (p + 1))
        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)

        return min(100 * h0, h1)

    def get_initial_ktarget(self) -> int:
        """very rough estimate from numerical recipes, can be motivated for example from Hairer&Wanner II, Fig.4.1"""

        log_fact = -np.log10(self.rtol + self.atol) * 0.6 + 0.5
        return int(log_fact)

    def get_initial_ktarget_sym(self) -> int:
        """very rough estimate from numerical recipes, can be motivated for example from Hairer&Wanner II, Fig.4.1"""

        log_fact = -max(-12.0, np.log10(self.rtol)) * 0.6 + 0.5
        return int(log_fact)

    def get_scaled_error_norm(
        self,
        error: NDArray[np.floating],
        x_ref: NDArray[np.floating],
    ) -> float:
        tol = self.atol + self.rtol * x_ref
        error_ratio = self.norm(error / tol)
        return error_ratio
