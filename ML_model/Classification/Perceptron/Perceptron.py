import numpy as np
from ML_model.Classification.Perceptron.activation_function import unit_step_func


class Perceptron:
    def __init__(self, learning_rate: int, n_iters: int) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weight = None
        self.bias = None

    def fit(self, X: np.array, y: np.array) -> None:

        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        # convert labels to binary 0/1 based on positive threshold
        y_binary = np.array([1 if y_sample > 0 else 0 for y_sample in y])

        for _ in range(self.n_iters):
            for index, x in enumerate(X):
                linear_output = np.dot(x, self.weight) + self.bias
                y_pred = self.activation_func(linear_output)

                error = y_binary[index] - y_pred
                update = error * self.learning_rate

                self.weight += update * x
                self.bias += update

    def predict(self, X: np.array) -> np.array:
        return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.array) -> np.array:
        linear_output = np.dot(x, self.weight) + self.bias
        return self.activation_func(linear_output)
