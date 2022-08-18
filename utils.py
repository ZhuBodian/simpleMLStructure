import sys
import os
import tarfile
import urllib.request
import pandas as pd
from sklearn.datasets import fetch_openml
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def save_fig(path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) # 返回dataframe类型


def show_feature_coef(coef, text_labels, bottom, title):
    x = np.arange(1, 1 + len(text_labels))

    plt.figure()
    plt.bar(x, coef, width=0.5)
    plt.xlabel('Features')
    plt.title('Corresponding coefficient size of each feature')
    plt.xticks(x, text_labels, rotation=90)
    plt.subplots_adjust(bottom=bottom)
    save_fig('lin_reg_coef')
    plt.show()


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def get_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, data_home='.\datasets\mnist_784')
    mnist.keys()
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    return X, y, X_train, X_test, y_train, y_test


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


# 绘制所有可能的阈值下的精度与召回率图
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    # 两条线画一张图里
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown


# 画精度-召回率函数图
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown


def plot_cm(cm, text_label=None, process_cm=False):
    if process_cm:
        # 获得分类错误率，而不是错误的绝对值
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_conf_mx = cm / row_sums

        # 用0填充对角线，仅保留错误，重新画图
        np.fill_diagonal(norm_conf_mx, 0)
        # 行代表实际类，列代表预测类，第8列看起来很亮，说明许多图片被错误分类为8
        # PS：第8行不那么差，说明数字8被正确分类了，且注意到错误不完全对称
        # 说明精力可以花在改进8的错误上（如进一步搜集8的数据，或者添加特征写个算法统计闭环数量）
        cm = norm_conf_mx

    if text_label is not None:
        cm = pd.DataFrame(cm, columns=text_label, index=text_label)

    plt.figure()
    sns.heatmap(cm, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    plt.xlabel('True class')
    plt.ylabel('Predict class')


def plot_confusion_image(X_train, y_train, y_train_pred, cl_a, cl_b):
    # 分析单个错误也可以帮助获得洞见
    # 通过画图分析35正确分类，与错误分类的图像，获得洞见
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8, 8))

    plt.subplot(221)
    plot_digits(X_aa[:25], images_per_row=5)
    plt.title(f'True {cl_a}, Predict{cl_a}')

    plt.subplot(222)
    plot_digits(X_ab[:25], images_per_row=5)
    plt.title(f'True {cl_a}, Predict{cl_b}')

    plt.subplot(223)
    plot_digits(X_ba[:25], images_per_row=5)
    plt.title(f'True {cl_b}, Predict{cl_a}')

    plt.subplot(224)
    plot_digits(X_bb[:25], images_per_row=5)
    plt.title(f'True {cl_b}, Predict{cl_b}')
