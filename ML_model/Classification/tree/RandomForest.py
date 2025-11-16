import numpy as np
from ML_model.Classification.tree.utils import most_common_label, bootstrap_sample
from ML_model.Classification.tree.DecisionTree import DecisionTree


class RandomForest:
    def __init__(
        self, n_tree: int, min_samples_split: int, max_depht: int, n_feats: int = None
    ) -> None:

        self.n_tree = n_tree
        self.min_samples_split = min_samples_split
        self.max_depht = max_depht
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X: np.array, y: np.array) -> None:

        for _ in range(self.n_tree):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depht,
                n_feats=self.n_feats,
            )

            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: np.array) -> np.array:

        tree_preds = [tree.predict(X) for tree in self.trees]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]

        return np.array(y_pred)
