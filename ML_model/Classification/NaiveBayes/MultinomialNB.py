import numpy as np


class MultinomialNB:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def fit(self, X: np.array, y: np.array) -> None:

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        self.n_classes = len(self._classes)

        self._class_count = np.zeros(self.n_classes, dtype=np.float64)
        self._feature_count = np.zeros((self.n_classes, n_features), dtype=np.float64)

        for index, _class in enumerate(self._classes):
            X_class = X[y == _class]
            self._class_count[index] = X_class.shape[0]
            self._feature_count[index, :] = X_class.sum(axis=0)

        self._class_log_prior = np.log(self._class_count / n_samples)

        self._feature_log_prior = np.log(
            (self._feature_count + self.alpha)
            / (self._feature_count.sum(axis=1, keepdims=True) + self.alpha * n_features)
        )

    def predict(self, X: np.array) -> np.array:
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, X: np.array) -> int:
        log_probs = []

        for index, _class in enumerate(self._classes):
            log_prob = self._class_log_prior[index]
            log_prob += np.sum(X * self._feature_log_prior[index, :])
            log_probs.append(log_prob)

        return self._classes[np.argmax(log_probs)]
