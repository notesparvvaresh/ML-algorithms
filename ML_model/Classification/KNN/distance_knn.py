import numpy as np
from collections import Counter

from metrics import *


class DistanceKnn:
    def __init__(self, K: int, metric) -> None:
        self.K = K
        self.metric = metric

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def train(self, X_test: np.array) -> np.array:
        # kept method name 'train' for backward compatibility
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x: np.array) -> np.array:
        distance = [self.metric(x, x_train) for x_train in self.X_train]
        index_K_Nearest_Neighbors = np.argsort(distance)[: self.K]

        labels = {}

        for index in index_K_Nearest_Neighbors:
            key = self.y_train[index]
            w = 1.0 / (distance[index] + 1e-12)
            if key in labels:
                labels[key] += w
            else:
                labels[key] = w

        label, weight = max(labels.items(), key=lambda item: item[1])
        return label
