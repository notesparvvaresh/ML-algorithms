import numpy as np


class LassoRegression:
    def __init__(self, iter: int, learning_rate: float, lambda_param: float) -> None:
        self.iter = iter
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.costs = []

    def fit(self, X: np.array, y: np.array) -> None:
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            y_preidct = np.dot(X, self.weights) + self.bias

            loss = y_preidct - y

            cost = (np.sum(loss**2) / (2 * n_samples)) + self.lambda_param * np.sum(
                np.abs(self.weights)
            )
            self.costs.append(cost)

            dw = (np.dot(X.T, loss) / n_samples) + self.lambda_param * np.sign(
                self.weights
            )
            db = np.sum(loss) / n_samples

            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate

    def predict(self, X: np.array) -> np.array:
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
