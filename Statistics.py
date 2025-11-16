import numpy as np


def covariance(X1: np.array, X2: np.array) -> np.array:
    mean_X1 = np.mean(X1)
    mean_X2 = np.mean(X2)

    N = len(X1)

    covariance = np.sum((X1 - mean_X1) * (X2 - mean_X2)) / (N - 1)
    return covariance


def correlation(X1: np.array, X2: np.array) -> np.array:

    _covariance = covariance(X1, X2)

    std_X1 = np.std(X1, ddof=1)
    std_X2 = np.std(X2, ddof=1)

    return _covariance / (std_X1 * std_X2)


def cov_matrix(X: np.array) -> np.array:
    means = np.mean(X, axis=0)
    centered_data = X - means
    n = X.shape[0]

    cov_matrix = np.dot(centered_data.T, centered_data) / (n - 1)
    return cov_matrix
