import numpy as np


def Euclidean_Distance(x1: np.array, x2: np.array) -> np.array:
    return np.sqrt(np.sum((x1 - x2) ** 2))


def Manhattan_Distance(x1: np.array, x2: np.array) -> np.array:
    return np.sum(np.abs(x1 - x2))
