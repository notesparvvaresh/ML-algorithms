import numpy as np


def MSE(y: np.array, y_predict: np.array) -> np.array:
    n_samples = y.shape[0]
    return np.sum((y - y_predict) ** 2) / n_samples


def MAE(y: np.array, y_predict: np.array) -> np.array:
    n_samples = y.shape[0]
    return np.sum(np.abs(y - y_predict)) / n_samples


def R2(y: np.array, y_predict: np.array) -> np.array:
    y_mean = np.mean(y)
    return 1 - ((np.sum((y - y_predict) ** 2)) / (np.sum((y - y_mean) ** 2)))
