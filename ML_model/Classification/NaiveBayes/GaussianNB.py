import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing: float) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: np.array, y: np.array) -> None:

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        self.n_classes = len(self._classes)

        self._mean = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self._prios = np.zeros((self.n_classes), dtype=np.float64)

        for index, _class in enumerate(self._classes):
            X_class = X[y == _class]
            self._mean[index, :] = X_class.mean(axis=0)
            self._var[index, :] = X_class.var(axis=0)
            self._prios[index] = X_class.shape[0] / float(n_samples)

    def predict(self, X: np.array) -> np.array:
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x: np.array):
        posteriors = []

        for index, _class in enumerate(self._classes):
            prior = np.log(self._prios[index])
            posterior_sum = np.sum(np.log(self._pdf(index, x)))
            posteriors.append(posterior_sum + prior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, index: int, x: np.array) -> np.array:
        mean = self._mean[index]
        var = self._var[index] + self.var_smoothing

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
