from sklearn.svm import SVC
import sys
import sklearn
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
PROJECT_ROOT_DIR = "."
FOLDER_ID = "svr"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", FOLDER_ID)


def linear_svr_regression():
    def linear():
        iris = datasets.load_iris()
        X = iris["data"][:, (2, 3)]  # petal length, petal width
        # iris本身是三类
        y = iris["target"]

        # 取出前两类的数据
        setosa_or_versicolor = (y == 0) | (y == 1)
        X = X[setosa_or_versicolor]
        y = y[setosa_or_versicolor]

        # SVM Classifier model
        # c控制正则化程度，c越小，正则化程度越小，间隔越大，泛化程度越好；
        # c越大，正则化程度越大，间隔越小，泛化程度越差；
        svm_clf = SVC(kernel="linear", C=float("inf"))
        svm_clf.fit(X, y)

    def soft_marge():
        iris = datasets.load_iris()
        X = iris["data"][:, (2, 3)]  # petal length, petal width
        y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

        # 正则化参数c=1的模型，5-4左图
        svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
        ])

        svm_clf.fit(X, y)

    linear()
    soft_marge()


def non_linear_regression():
    def plot_dataset(X, y, axes):
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

    def ploy_kernal():
        # 参数kernel="poly", degree=3,coef0=1控制添加了多项式核
        # coef0控制的是模型受高阶多项式还是低阶多项式影响的程度
        poly_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
        ])
        poly_kernel_svm_clf.fit(X, y)

    def rbf_kernel():
        rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
        ])
        rbf_kernel_svm_clf.fit(X, y)


    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()







if __name__ == '__main__':
    np.random.rand(42)

    linear_svr_regression()
