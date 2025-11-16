import numpy as np


def get_info(y: np.array, y_pred: np.array) -> list:
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y: np.array, y_pred: np.array) -> float:
    tp, tn, fp, fn = get_info(y, y_pred)

    return (tp + tn) / (tp + tn + fp + fn)


def recall(y: np.array, y_pred: np.array) -> float:
    tp, tn, fp, fn = get_info(y, y_pred)
    return tp / (tp + fn)


def precision(y: np.array, y_pred: np.array) -> float:
    tp, tn, fp, fn = get_info(y, y_pred)
    return tp / (tp + fp)


def f1score(y: np.array, y_pred: np.array) -> float:
    _recall = recall(y, y_pred)
    _precision = precision(y, y_pred)
    return (2 * (_precision * _recall)) / (_precision + _recall)
