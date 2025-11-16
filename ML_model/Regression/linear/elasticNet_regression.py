import numpy as np


class LinearRegression:
    def __init__(
        self,
        iter: int,
        learning_rate: float,
        l1_pharameter: float,
        l2_pharameter: float,
    ) -> None:
        self.iter = iter
        self.learning_rate = learning_rate
        self.l1_pharameter = l1_pharameter
        self.l2_pharameter = l2_pharameter
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

            l1_cost = self.l1_pharameter * np.sum(np.abs(self.weights))
            l2_cost = (self.l2_pharameter / 2) * np.sum(self.weights**2)

            cost = np.sum(loss**2) / (2 * n_samples) + l1_cost + l2_cost
            self.costs.append(cost)

            l1_dw = self.l1_pharameter * np.sign(self.weights)
            l2_dw = self.l2_pharameter * self.weights
            dw = (np.dot(X.T, loss) / n_samples) + l1_dw + l2_dw
            db = np.sum(loss) / n_samples

            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate

    def train(self, X: np.array) -> np.array:
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
