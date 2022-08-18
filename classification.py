import sys
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
import utils
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
PROJECT_ROOT_DIR = "."
FOLDER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", FOLDER_ID)


def display_mnist(X):
    # 维度为1*784，而不是1*1
    some_digit = X[0]
    some_digit_image = some_digit.reshape(28, 28)
    # 画热图，cmp选择灰度binary
    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")

    utils.save_fig(IMAGES_PATH, "some_digit_plot")
    plt.show()

    plt.figure(figsize=(9, 9))
    example_images = X[:100]  # 前100行数据
    utils.plot_digits(example_images, images_per_row=10)
    utils.save_fig(IMAGES_PATH, "more_digits_plot")
    plt.show()


def training_a_binary_classifier(X_train, y_train, y_test):
    def cv_score():
        print('CV SCORE'.center(100, '*'))
        print('交叉验证计算rmse'.center(75, '*'))
        scores = cross_val_score(sgd_clf, X_train, y_train_5.astype(np.uint8), cv=3, scoring="neg_mean_squared_error",
                                 verbose=10, n_jobs=-1)
        sgd_rmse_scores = np.sqrt(-scores)
        print(f'交叉验证决策树回归损失：')
        utils.display_scores(sgd_rmse_scores)

    def cv_predict():
        print('交叉验证进行预测'.center(75, '*'))
        # 注意虽然是三折交叉验证，但是最终返回的却只有一次预测（每个折叠对策）
        y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, verbose=10)
        print('混淆矩阵：')
        print(confusion_matrix(y_train_5, y_train_pred))
        print('准确率：')
        print(precision_score(y_train_5, y_train_pred))
        print('召回率：')
        print(recall_score(y_train_5, y_train_pred))
        print('F1分数：')
        print(f1_score(y_train_5, y_train_pred))

        """画混淆矩阵"""
        text_label = ['not 5', '5']
        cm = confusion_matrix(y_train_5, y_train_pred)
        utils.plot_cm(cm, text_label)
        utils.save_fig(IMAGES_PATH, '二类混淆矩阵')
        plt.show()

    def performance_measure():
        # 获得所有实例的分数，为将来遍历所有可能阈值做准备
        y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function", verbose=10)
        precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

        # 由图中可以看出，精度在90%上升的速度变慢，可以考虑取精度为90%时的阈值为所要求的阈值
        # np.argmax会返回最大值的第一个索引
        # 求该阈值下的召回率，布尔索引
        target_precision = 0.9
        recall_90_precision = recalls[np.argmax(precisions >= target_precision)]
        # 求该阈值的具体值
        threshold_90_precision = thresholds[np.argmax(precisions >= target_precision)]

        """精度召回率随阈值变化曲线"""
        plt.figure(figsize=(8, 4))  # Not shown
        utils.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        plt.plot([threshold_90_precision, threshold_90_precision], [0., target_precision], "r:")  # Not shown
        plt.plot([-50000, threshold_90_precision], [target_precision, target_precision], "r:")  # Not shown
        plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")  # Not shown
        plt.plot([threshold_90_precision], [target_precision], "ro")  # Not shown
        plt.plot([threshold_90_precision], [recall_90_precision], "ro")  # Not shown
        utils.save_fig(IMAGES_PATH ,"precision_recall_vs_threshold_plot")  # Not shown
        plt.show()
        # 注意提高阈值时，精度有时也可能会下降，尽管总体上是上升，所以会陡峭一点
        # 而阈值提升时，召回率只会下降，所以要平滑一点

        """精度-召回率变化曲线"""
        plt.figure(figsize=(8, 6))
        utils.plot_precision_vs_recall(precisions, recalls)
        plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
        plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
        plt.plot([recall_90_precision], [0.9], "ro")
        utils.save_fig(IMAGES_PATH ,"precision_vs_recall_plot")
        plt.show()

        """ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
        plt.figure(figsize=(8, 6))  # Not shown
        utils.plot_roc_curve(fpr, tpr)
        fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]  # Not shown
        plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")  # Not shown
        plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
        plt.plot([fpr_90], [recall_90_precision], "ro")  # Not shown
        utils.save_fig(IMAGES_PATH, "roc_curve_plot")  # Not shown
        plt.show()

    # 二类分类器，其标签只需是0或者1即可（布尔类型表示）
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

    # cv_score()
    # cv_predict()
    # performance_measure()


def multi_classification(X_train, y_train):
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3, verbose=10)

    cm = confusion_matrix(y_train, y_train_pred)
    text_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    utils.plot_cm(cm, text_label)
    plt.show()
    utils.save_fig(IMAGES_PATH, '混淆矩阵')

    utils.plot_cm(cm, text_label, process_cm=True)
    plt.show()
    utils.save_fig(IMAGES_PATH, '处理后的混淆矩阵')

    utils.plot_confusion_image(X_train, y_train, y_train_pred, 3, 5)
    plt.show()
    utils.save_fig(IMAGES_PATH, '3与5混淆图')



if __name__ == '__main__':
    np.random.seed(42)
    X, y, X_train, X_test, y_train, y_test = utils.get_mnist()

    # display_mnist(X)
    # training_a_binary_classifier(X_train, y_train, y_test)
    multi_classification(X_train, y_train)

