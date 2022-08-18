import sys
import sklearn
import os
from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import utils

assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
PROJECT_ROOT_DIR = "."
FOLDER_ID = "ensemble_learning_and_random_forests"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", FOLDER_ID)


def voting_classifier(X_train, X_test, y_train, y_test):
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()

    # 硬投票,voting='soft'即为软投票
    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
    voting_clf.fit(X_train, y_train)

    # 可以看出来投票分类器略胜于所有单个分类器
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


def bagging_and_pasting(X_train, X_test, y_train, y_test):
    def bagging():
        # 训练了一个包含500个决策树分类器的集成，每次从训练集中随机采样100个训练实例进行训练，然后放回（bagging）
        # 如果想pasting，可以设置bootstrap=False
        bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True)
        bag_clf.fit(X_train, y_train)
        y_pred = bag_clf.predict(X_test)

        print(accuracy_score(y_test, y_pred))

    def out_of_bag_evaluation():
        bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,bootstrap=True, oob_score=True)
        bag_clf.fit(X_train, y_train)
        # 查看包外评估分数
        bag_clf.oob_score_

        # 返回每个包外实例，各类的决策概率
        bag_clf.oob_decision_function_[:5]

    bagging()
    out_of_bag_evaluation()


def random_forests(X_train, y_train):
    """A Random Forest is equivalent to a bag of decision trees:
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),n_estimators=500)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    """
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)

    feature_names = ['x_0', 'x_1']
    for name, score in zip(feature_names, rnd_clf.feature_importances_):
        print(name, score)
    # 展示随机森林获取的，各个特征的相对重要性


def my_boosting():
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R",
                                 learning_rate=0.5)
    ada_clf.fit(X_train, y_train)


def my_gradient_boosting_tree(X_train, X_val, y_train, y_val):
    def my_plot():
        plt.figure(figsize=(8, 8))

        plt.plot(errors, "b.-")
        plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
        plt.plot([0, 120], [min_error, min_error], "k--")
        plt.plot(bst_n_estimators, min_error, "ko")
        plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="center", fontsize=14)
        # plt.axis([0, 120, 0, 0.01])
        plt.xlabel("Number of trees")
        plt.ylabel("Error", fontsize=16)
        plt.title("Validation error", fontsize=14)

        utils.save_fig(IMAGES_PATH, "early_stopping_gbrt_plot")
        plt.show()

    # 训练拥有120棵树的GBRT集成
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42, verbose=10)
    gbrt.fit(X_train, y_train)

    # 测量每个训练阶段的验证误差，从而找到树的最优数量（验证误差最小的模型的树）
    # staged_predict方法在训练的每个阶段（一棵树、两棵树时，等等）都对集成的预测返回一个迭代器
    errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
    # 树数从1开始，errors为list列表，索引从0开始
    bst_n_estimators = np.argmin(errors) + 1

    # 用最优树数重新训练一个GBRT集成
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)

    min_error = np.min(errors)
    my_plot()


if __name__ == '__main__':
    np.random.rand(42)
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # voting_classifier(X_train, X_test, y_train, y_test)
    # bagging_and_pasting(X_train, X_test, y_train, y_test)
    # random_forests(X_train, y_train)
    # my_boosting(X_train, X_test, y_train, y_test)
    my_gradient_boosting_tree(X_train, X_test, y_train, y_test)