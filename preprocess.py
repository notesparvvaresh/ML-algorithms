import numpy as np


class StandardScaler:
    def __init__(self) -> None:
        self._mean = None
        self._std = None

    def fit(self, X: np.array) -> None:
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0, ddof=0)

    def transform(self, X) -> np.array:

        if self._mean is None or self._std is None:
            raise Exception("The scaler has not been fitted yet!")

        return (X - self._mean) / self._std


class MinMaxScaler:
    def __init__(self) -> None:
        self._min = None
        self._max = None

    def fit(self, X: np.array) -> None:
        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)

    def transform(self, X: np.array) -> np.array:
        if self._min is None or self._max is None:
            raise Exception("The scaler has not been fitted yet!")

        return (X - self._min) / (self._max - self._min)


class MaxAbsScaler:
    def __init__(self):
        self._max_abs = None

    def fit(self, X: np.array) -> None:
        self._max_abs = np.max(np.abs(X), axis=0)

    def transform(self, X: np.array) -> np.array:
        if self._max_abs is None:
            raise Exception("The scaler has not been fitted yet!")
        return X / self._max_abs


class RobustScaler:
    def __init__(self):
        self._median = None
        self._iqr = None

    def fit(self, X: np.array) -> None:
        self._median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self._iqr = q75 - q25

    def transform(self, X: np.array) -> np.array:
        if self._median is None or self._iqr is None:
            raise Exception("The scaler has not been fitted yet!")
        return (X - self._median) / self._iqr


class Normalizer:
    def __init__(self, norm="l2"):
        self.norm = norm

    def fit(self, X: np.array) -> None:
        if self.norm == "l2":
            self.norms = np.linalg.norm(X, axis=1, keepdims=True)
            self.norms[self.norms == 0] = 1

        elif self.norm == "l1":
            self.norms = np.sum(np.abs(X), axis=1, keepdims=True)
            self.norms[self.norms == 0] = 1
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

    def transform(self, X) -> np.array:
        if self.norms is None or self.norm is None:
            raise Exception("The scaler has not been fitted yet!")
        return X / self.norms
