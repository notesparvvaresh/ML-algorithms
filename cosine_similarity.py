import numpy as np


def cosine_similarity(a: np.array, b: np.array) -> np.array:
    dot_product = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity


class CosineSimilarityMatrix:
    def __init__(self):
        self._cosine_similarity_matrix = None

    def fit(self, X: np.array) -> None:
        self.X = X
        n_samples = X.shape[0]
        self._cosine_similarity_matrix = np.zeros((n_samples, n_samples))

    def transform(self) -> np.array:
        if self._cosine_similarity_matrix is None:
            raise Exception("The cosine similarity matrix has not been fitted yet!")

        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        normalized_data = self.X / norms

        self._cosine_similarity_matrix = np.dot(normalized_data, normalized_data.T)

        return self._cosine_similarity_matrix
