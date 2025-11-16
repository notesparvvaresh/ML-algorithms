import numpy as np


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ) -> None:

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None
