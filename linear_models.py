import sys
import sklearn
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import utils
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
PROJECT_ROOT_DIR = "."
FOLDER_ID = "linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", FOLDER_ID)


def polynomial_regression():
    def my_plot():
        X_new = np.linspace(-3, 3, 100).reshape(100, 1)
        X_new_poly = poly_features.transform(X_new)
        y_new = lin_reg.predict(X_new_poly)
        plt.plot(X, y, "b.")
        plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.legend(loc="upper left", fontsize=14)
        plt.axis([-3, 3, 0, 10])
        utils.save_fig(IMAGES_PATH, "quadratic_predictions_plot")
        plt.show()
    # sklearn如何多项式回归：广义线性回归（没有直接多项式回归的函数， 但是有sklearn.preprocessing.PolynomialFeatures）
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    # x_poly包含了原始特征及其平方
    X_poly = poly_features.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    my_plot()


def regularized_linear_models():
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)

    # 采用cholesky（矩阵分解技术）进行岭回归
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)

    # 在SGD中采用l2正则项（其实也是默认选项），也相当于岭回归
    sgd_reg = SGDRegressor(penalty="l2", random_state=42)
    sgd_reg.fit(X, y.ravel())

    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)

    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X, y)


def logistic_regression():
    iris = datasets.load_iris()

    # 仅选择一个特征用于预测
    X = iris["data"][:, 3:]  # petal width
    # 多类转化为2类，即是与非
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

    log_reg = LogisticRegression()
    log_reg.fit(X, y)


def soft_regression():
    iris = datasets.load_iris()

    # 选择两个预测特征
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    # 转化为二分类问题
    y = (iris["target"] == 2).astype(np.int)

    log_reg = LogisticRegression(solver="lbfgs", C=10 ** 10)
    log_reg.fit(X, y)

    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)





if __name__ == '__main__':
    np.random.rand(42)
    polynomial_regression()
    regularized_linear_models()
    logistic_regression()

