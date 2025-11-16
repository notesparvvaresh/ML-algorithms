import numpy as np
from Activator_function import sigmoid


class Logsticregression:
    def __init__(self, learning_rate: float, n_iters: int) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.costs = None

    def fit(self, X: np.array, y: np.array) -> None:
        n_samples, n_features = X.shape
        self.costs = []

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predict = sigmoid(linear_model)

            self.costs.append(self._compute_cost(y, y_predict))
            error = y_predict - y

            dw = np.dot(X.T, error) / (n_samples)
            db = np.sum(error) / (n_samples)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.array) -> np.array:
        linear_model = np.dot(X.T, self.weights) + self.bias
        y_predict = sigmoid(linear_model)
        y_predict = [1 if y_pred > 0.5 else 0 for y_pred in y_predict]
        return np.array(y_predict)

    def _compute_cost(self, y: np.array, y_pred: np.array) -> np.float_:
        n_sampels = len(y)

        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        cost = -(np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))) / (
            n_sampels
        )
        return cost
