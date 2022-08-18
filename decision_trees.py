import sys
import sklearn
import os
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.datasets import make_moons


assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
PROJECT_ROOT_DIR = "."
FOLDER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", FOLDER_ID)


def training_and_visualizing_a_decision_tree():
    iris = load_iris()
    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, y)

    export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

    Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))

    # 估计特定类的概率
    tree_clf.predict_proba([[5, 1.5]])

    # 仅输出最可能的类
    tree_clf.predict([[5, 1.5]])


def regularization_hyperparameters():
    iris = load_iris()
    X = iris.data[:, 2:]  # petal length and width
    y = iris.target
    tree_clf_tweaked = DecisionTreeClassifier(max_depth=2)
    tree_clf_tweaked.fit(X, y)

    Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)
    deep_tree_clf = DecisionTreeClassifier(min_samples_leaf=4)
    deep_tree_clf.fit(Xm, ym)


if __name__ == '__main__':
    np.random.rand(42)

    training_and_visualizing_a_decision_tree()
    regularization_hyperparameters()