import numpy as np

from ML_model.Classification.tree.node import Node
from ML_model.Classification.tree.utils import entropy, most_common_label


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X: np.array, y: np.array) -> None:
        # store a fallback label for empty child splits
        self._most_common_label = most_common_label(y)
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X: np.array):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.array, y: np.array, depht: int = 0) -> Node:

        # handle empty input (possible from bootstrap/resampling)
        if X.size == 0:
            return Node(value=self._most_common_label)

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depht >= self.max_depth
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            # avoid calling most_common_label on empty y
            leaf_value = most_common_label(y) if n_samples > 0 else self._most_common_label
            return Node(value=leaf_value)

        features_index = np.random.choice(n_features, self.n_feats, replace=True)
        best_feature, best_threshold = self._best_split(X, y, features_index)

        left_index, right_index = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_index, :], y[left_index], depht + 1)
        right = self._grow_tree(X[right_index, :], y[right_index], depht + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X: np.array, y: np.array, features_index: np.array) -> list:

        best_gain = -1
        split_index, split_threshod = None, None

        for feature_index in features_index:
            X_column = X[:, feature_index]
            threshods = np.unique(X_column)
            for threshod in threshods:
                gain = self._information_gain(y, X_column, threshod)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshod = threshod

        return split_index, split_threshod

    def _information_gain(
        self, y: np.array, X_column: np.array, threshod: int
    ) -> float:
        parent_entropy = entropy(y)

        left_index, right_index = self._split(X_column, threshod)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        n = len(y)
        len_left, len_right = len(left_index), len(right_index)
        en_left, en_right = entropy(y[left_index]), entropy(y[right_index])
        child_entropy = (len_left / n) * en_left + (len_right / n) * en_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X: np.array, threshod: int) -> list:
        left_index = np.argwhere(X <= threshod).flatten()
        right_index = np.argwhere(X > threshod).flatten()
        return left_index, right_index

    def _traverse_tree(self, x: np.array, node: Node) -> np.array:
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
