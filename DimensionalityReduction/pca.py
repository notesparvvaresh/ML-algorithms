import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        if self.n_components is not None:
            eigvecs = eigvecs[:, : self.n_components]
        self.components_ = eigvecs

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
