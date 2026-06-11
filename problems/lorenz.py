import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from solvers.explicit import *
from solvers.Extrapolation_Scheme import (
    EulerExtrapolation,
)
import logging

logging.basicConfig(level=logging.DEBUG)

# Lorenz System
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
def x_dot(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.array(
        [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    )
t_max = 100.0
x0 = np.array([1.0, 1.0, 1.0])

s = EulerExtrapolation(ode_fun=x_dot)
time, result, solve_info = s.solve(x0, t_max, 1e-3)

fig, ax = plt.subplots()
# ax.plot(result[0], result[1])
ax.plot(time, result[:, 0], marker="o")
plt.show()
