import numpy as np
from collections import Counter


def entropy(y: np.array) -> np.array:
    hist = np.bincount(y)
    ps = hist / len(y)

    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def most_common_label(y: np.array) -> int:
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


def bootstrap_sample(X: np.array, y: np.array) -> list:
    n_sample = X.shape[0]
    index_bootstrap_sample = np.random.choice(n_sample, n_sample, replace=True)
    return X[index_bootstrap_sample], y[index_bootstrap_sample]
