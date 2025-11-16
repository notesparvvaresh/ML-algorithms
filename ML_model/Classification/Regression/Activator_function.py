import numpy as np


def sigmoid(X: np.array) -> np.array:
    return 1 / (1 + np.exp(-X))
