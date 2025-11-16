import numpy as np


class LinearRegression:
    def __init__(self, iter: int, learning_rate: float) -> None:
        self.iter = iter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.costs = []

    def fit(self, X: np.array, y: np.array) -> None:
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            y_predict = np.dot(X, self.weights) + self.bias
            loss = y_predict - y

            cost = np.sum(loss**2) / (2 * n_samples)
            self.costs.append(cost)

            dw = np.dot(X.T, loss) / n_samples
            db = np.sum(loss) / n_samples

            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate

    def train(self, X: np.array) -> np.array:
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
