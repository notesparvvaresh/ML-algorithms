import numpy as np


def unit_step_func(x: np.array) -> np.array:
    return np.where(x >= 0, 1, 0)
