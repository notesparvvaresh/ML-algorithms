import sys
import os
import traceback
import numpy as np

# Ensure repo root is on path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = []


def check(name, func):
    try:
        func()
        print(f"[PASS] {name}")
        RESULTS.append((name, True, ""))
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback_str = traceback.format_exc()
        RESULTS.append((name, False, traceback_str))


# Tests


def test_preprocess():
    from preprocess import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    ss = StandardScaler()
    ss.fit(X)
    out = ss.transform(X)
    assert out.shape == X.shape

    mm = MinMaxScaler()
    mm.fit(X)
    out2 = mm.transform(X)
    assert out2.min() >= 0 and out2.max() <= 1

    rs = RobustScaler()
    rs.fit(X)
    out3 = rs.transform(X)
    assert out3.shape == X.shape

    n = Normalizer("l2")
    n.fit(X)
    out4 = n.transform(X)
    assert np.allclose(np.linalg.norm(out4, axis=1), 1)


def test_cosine():
    from cosine_similarity import cosine_similarity, CosineSimilarityMatrix

    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 0.0)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    csm = CosineSimilarityMatrix()
    csm.fit(X)
    mat = csm.transform()
    assert mat.shape == (2, 2)


def test_knn_classification():
    from ML_model.Classification.KNN.uniform_knn import UniformKnn
    from ML_model.Classification.KNN.distance_knn import DistanceKnn

    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [1.1, 1.1]])
    y = np.array([0, 1, 0, 1])
    uk = UniformKnn(K=3, metric=lambda x, y: np.linalg.norm(x - y))
    uk.fit(X, y)
    preds = uk.predict(X)
    assert preds.shape == y.shape

    dk = DistanceKnn(K=3, metric=lambda x, y: np.linalg.norm(x - y))
    dk.fit(X, y)
    preds2 = dk.train(X)
    assert preds2.shape == y.shape


def test_knn_regression():
    from ML_model.Regression.KNN.uniform_knn import UniformKnn as RegUniform
    from ML_model.Regression.KNN.distance_knn import DistanceKnn as RegDistance

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    uk = RegUniform(K=3, metric=lambda x, y: np.linalg.norm(x - y))
    uk.fit(X, y)
    preds = uk.predict(X)
    assert preds.shape == y.shape

    dk = RegDistance(K=3, metric=lambda x, y: np.linalg.norm(x - y))
    dk.fit(X, y)
    preds2 = dk.train(X)
    assert preds2.shape == y.shape


def test_perceptron():
    from ML_model.Classification.Perceptron.Perceptron import Perceptron

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    p = Perceptron(learning_rate=0.1, n_iters=10)
    p.fit(X, y)
    preds = p.predict(X)
    assert preds.shape == y.shape


def test_naive_bayes():
    from ML_model.Classification.NaiveBayes.GaussianNB import GaussianNB
    from ML_model.Classification.NaiveBayes.MultinomialNB import MultinomialNB

    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [1.1, 1.1]])
    y = np.array([0, 1, 0, 1])
    g = GaussianNB(var_smoothing=1e-9)
    g.fit(X, y)
    preds = g.predict(X)
    assert preds.shape == y.shape

    Xm = np.array([[2, 0, 1], [1, 0, 2], [0, 1, 1], [0, 2, 0]])
    ym = np.array([0, 0, 1, 1])
    m = MultinomialNB(alpha=1.0)
    m.fit(Xm, ym)
    preds2 = m.predict(Xm)
    assert preds2.shape == ym.shape


def test_tree_models():
    from ML_model.Classification.tree.DecisionTree import DecisionTree
    from ML_model.Classification.tree.RandomForest import RandomForest
    from ML_model.Classification.tree.Adaboost import AdaBoost

    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    preds = dt.predict(X)
    assert preds.shape == y.shape

    rf = RandomForest(n_tree=3, min_samples_split=2, max_depht=3)
    rf.fit(X, y)
    preds_rf = rf.predict(X)
    assert preds_rf.shape == y.shape

    adb = AdaBoost(n_clf=3, learning_rate=1.0)
    adb.fit(X, y)
    preds_adb = adb.predict(X)
    assert preds_adb.shape == y.shape


def test_linear_regressions():
    from ML_model.Regression.linear.linear_regression import LinearRegression
    from ML_model.Regression.linear.ridge_regression import RidgeRegression
    from ML_model.Regression.linear.lasso_regression import LassoRegression
    from ML_model.Regression.linear.elasticNet_regression import (
        LinearRegression as ElasticNet,
    )

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    lr = LinearRegression(iter=200, learning_rate=0.01)
    lr.fit(X, y)
    preds = lr.train(X)
    assert preds.shape == y.shape

    rr = RidgeRegression(iter=200, learning_rate=0.01, alpha=0.1)
    rr.fit(X, y)
    preds2 = rr.train(X)
    assert preds2.shape == y.shape

    ls = LassoRegression(iter=200, learning_rate=0.01, lambda_param=0.1)
    ls.fit(X, y)
    preds3 = ls.predict(X)
    assert preds3.shape == y.shape

    en = ElasticNet(
        iter=200, learning_rate=0.01, l1_pharameter=0.01, l2_pharameter=0.01
    )
    en.fit(X, y)
    preds4 = en.train(X)
    assert preds4.shape == y.shape


def test_metrics():
    from metrics.classification import accuracy, precision, recall, f1score
    from metrics.regression import MSE, MAE, R2

    y = np.array([0, 1, 1, 0])
    yp = np.array([0, 1, 0, 0])
    _ = accuracy(y, yp)
    _ = precision(y, yp)
    _ = recall(y, yp)
    _ = f1score(y, yp)

    y_r = np.array([1.0, 2.0, 3.0])
    yp_r = np.array([1.1, 1.9, 3.0])
    _ = MSE(y_r, yp_r)
    _ = MAE(y_r, yp_r)
    _ = R2(y_r, yp_r)


def run_all():
    checks = [
        ("preprocess", test_preprocess),
        ("cosine", test_cosine),
        ("knn_classification", test_knn_classification),
        ("knn_regression", test_knn_regression),
        ("perceptron", test_perceptron),
        ("naive_bayes", test_naive_bayes),
        ("tree_models", test_tree_models),
        ("linear_regressions", test_linear_regressions),
        ("metrics", test_metrics),
    ]

    failures = 0
    for name, func in checks:
        check(name, func)
        if not RESULTS[-1][1]:
            failures += 1

    print("\nSUMMARY:")
    for name, ok, tb in RESULTS:
        print(f" - {name}: {'OK' if ok else 'FAIL'}")

    if failures:
        print(f"\n{failures} test groups failed.")
        sys.exit(2)
    else:
        print("\nAll test groups passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    run_all()
