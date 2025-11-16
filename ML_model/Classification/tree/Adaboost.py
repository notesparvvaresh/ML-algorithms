import numpy as np


class DecisionStump:
    def __init__(self) -> None:

        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X: np.array) -> np.array:
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_clf: int, learning_rate: float) -> None:
        self.n_clf = n_clf
        self.learning_rate = learning_rate

    def fit(self, X: np.array, y: np.array) -> None:
        n_samples, n_features = X.shape
        # convert labels to -1/1
        y_signed = np.where(y <= 0, -1, 1)

        w = np.full(n_samples, (1 / n_samples))
        self.clfs = list()

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            for feature_index in range(n_features):
                X_column = X[:, feature_index]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    miss_classified = w[y_signed != predictions]
                    error = np.sum(miss_classified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_index
                        min_error = error

            EPS = 1e-10
            clf.alpha = (
                self.learning_rate
                * 0.5
                * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            )
            predictions = clf.predict(X)

            # update sample weights (use signed labels -1/1)
            w *= np.exp(-clf.alpha * y_signed * predictions)

            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X: np.array) -> np.array:
        clfs_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clfs_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
