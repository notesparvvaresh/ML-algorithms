import sys
import os
import traceback

# Add repo root to path so imports work when running the script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np


def run():
    try:
        print("Running quick checks for core modules...\n")

        # Preprocess scalers
        from preprocess import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        ss = StandardScaler()
        ss.fit(X)
        print("StandardScaler transform:", ss.transform(X))

        mm = MinMaxScaler()
        mm.fit(X)
        print("MinMaxScaler transform:", mm.transform(X))

        rs = RobustScaler()
        rs.fit(X)
        print("RobustScaler transform:", rs.transform(X))

        norm = Normalizer("l2")
        norm.fit(X)
        print("Normalizer transform:", norm.transform(X))

        # Cosine similarity
        from cosine_similarity import cosine_similarity, CosineSimilarityMatrix

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        print("cosine(a,b):", cosine_similarity(a, b))
        csm = CosineSimilarityMatrix()
        csm.fit(X)
        print("Cosine matrix:\n", csm.transform())

        # KNN classification (UniformKnn)
        from ML_model.Classification.KNN.uniform_knn import UniformKnn
        from ML_model.Classification.KNN.distance_knn import (
            DistanceKnn as ClassDistanceKnn,
        )

        Xc = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [1.1, 1.1]])
        yc = np.array([0, 1, 0, 1])
        uk = UniformKnn(K=3, metric=lambda x, y: np.linalg.norm(x - y))
        uk.fit(Xc, yc)
        print("UniformKnn predict:", uk.predict(Xc))
        dk = ClassDistanceKnn(K=3, metric=lambda x, y: np.linalg.norm(x - y))
        dk.fit(Xc, yc)
        print("DistanceKnn (classification) train/predict:", dk.train(Xc))

        # Perceptron
        from ML_model.Classification.Perceptron.Perceptron import Perceptron

        Xp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        yp = np.array([0, 0, 0, 1])
        p = Perceptron(learning_rate=0.1, n_iters=10)
        p.fit(Xp, yp)
        print("Perceptron predict:", p.predict(Xp))

        # GaussianNB
        from ML_model.Classification.NaiveBayes.GaussianNB import GaussianNB

        g = GaussianNB(var_smoothing=1e-9)
        g.fit(Xc, yc)
        print("GaussianNB predict:", g.predict(Xc))

        # MultinomialNB (simple check)
        from ML_model.Classification.NaiveBayes.MultinomialNB import MultinomialNB

        # prepare count-like data for multinomial
        Xm = np.array([[2, 0, 1], [1, 0, 2], [0, 1, 1], [0, 2, 0]])
        ym = np.array([0, 0, 1, 1])
        mnb = MultinomialNB(alpha=1.0)
        mnb.fit(Xm, ym)
        print("MultinomialNB predict:", mnb.predict(Xm))

        # Linear Regression (small sanity check)
        from ML_model.Regression.linear.linear_regression import LinearRegression

        Xr = np.array([[1.0], [2.0], [3.0], [4.0]])
        yr = np.array([2.0, 4.0, 6.0, 8.0])
        lr = LinearRegression(iter=1000, learning_rate=0.01)
        lr.fit(Xr, yr)
        print("LinearRegression predict:", lr.train(Xr))

        print("\nALL QUICK CHECKS COMPLETED SUCCESSFULLY")
        return 0

    except Exception as e:
        print("ERROR during tests:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run())
