import numpy as np


class Minimum_distance:
    def __init__(self, shrink_threshold: float, metric) -> None:
        self.shrink_threshold = shrink_threshold
        self.metric = metric
        self.classes = None
        self.centroid = None

    def fit(self, X: np.array, y: np.array) -> None:

        self.classes = np.unique(y)
        self.centroid = list()

        self.global_mean = np.mean(X, axis=0)

        for index, _class in enumerate(self.classes):
            X_class = X[y == _class]
            center = np.mean(X_class, axis=0)

            if self.shrink_threshold is not None:
                feature_var = np.var(X_class, axis=0)
                shrink_indices = feature_var < self.shrink_threshold
                center[shrink_indices] = self.global_mean[shrink_indices]

            self.centroid.append(center)
        self.centroid = np.array(self.centroid)

    def predict(self, X: np.array) -> np.array:

        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x: np.array) -> np.array:
        distance = [self.metric(center, x) for center in self.centroid]
        index_neareset_center = np.argmin(distance)

        return self.classes[index_neareset_center]
