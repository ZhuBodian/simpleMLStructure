import sys
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, rooms_ix, bedrooms_ix, population_ix, households_ix, add_bedrooms_per_room=True):
        # 添加超参数add_bedrooms_per_room
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # 可以看出X应为ndarray类型
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]  # np.c_是按行连接两个矩阵
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def take_a_quick_look_at_the_data_structure():
    housing = utils.load_housing_data()
    print(f'housing.head()：')
    print(housing.head())

    print(f'housing.info()：')
    housing.info()

    print(f'housing["ocean_proximity"].value_counts()：')
    print(housing["ocean_proximity"].value_counts())

    print(f'housing.describe()：')
    print(housing.describe())

    housing.hist(bins=50, figsize=(20, 15))
    utils.save_fig("attribute_histogram_plots")
    plt.show()

    return housing


def create_a_test_set(housing):
    # 注意要把X，y合在一起
    train_set, test_set = train_test_split(housing, test_size=0.2)
    print("训练集数量：", len(train_set))
    print("测试集数量：", len(test_set))
    print("test_set.head():")
    print(test_set.head())

    housing["median_income"].hist()

    # 根据median_income分层，添加新列income_cat，该列的取值为1,2,3,4,5
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    print('housing["income_cat"].value_counts()：')
    print(housing["income_cat"].value_counts())

    housing["income_cat"].hist()

    # train_test_split是方法，直接返回train_set, test_set
    # StratifiedShuffleSplit为类
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    # sklearn.model_selection.StratifiedShuffleSplit.split()方法应该把X，y分开，且返回的索引
    # 本例未完全分开
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        # loc：Access a group of rows and columns by label(s) or a boolean array.
        # 用housing[train_index,:]是不行的，why?
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # 查看测试集中各类比例
    print('查看测试集中各类比例')
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    print('查看原数据集中各类的比例')
    print(housing["income_cat"].value_counts() / len(housing))

    # 训练集与测试集均删除这个新特征
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set


def discover_and_visualize_the_data_to_gain_insights(strat_train_set):
    # copy()为值传递。若直接赋值新的变量，那么其实还是引用传递
    housing = strat_train_set.copy()

    # 设置不透明度降低有助于帮助强调密度高的区域
    # 圈的大小由人口密度决定，颜色的不同由收入中位数值决定
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 sharex=False)
    plt.legend()
    utils.save_fig("housing_prices_scatterplot")

    import matplotlib.image as mpimg
    images_path = os.path.join(PROJECT_ROOT_DIR, "datasets", "housing")
    filename = "california.png"
    california_img = mpimg.imread(os.path.join(images_path, filename))
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                      s=housing['population'] / 100, label="Population",
                      c="median_house_value", cmap=plt.get_cmap("jet"),
                      colorbar=False, alpha=0.4)
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar(ticks=tick_values / prices.max())
    cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)

    plt.legend(fontsize=16)
    utils.save_fig("california_housing_prices_plot")
    plt.show()

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    # from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    utils.save_fig("scatter_matrix_plot")

    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    utils.save_fig("income_vs_house_value_scatterplot")

    # 通过组合特征添加新特征
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    # 新特征要比原特征与median_house_value的相关性更高
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
                 alpha=0.2)
    plt.axis([0, 5, 0, 520000])
    plt.show()

    print('housing.describe()：')
    print(housing.describe())


def prepare_the_data_for_machine_learning_algorithms(strat_train_set):
    # 把strat_train_set中的median_house_value属性删除后复制给housing，且drop不影响strat_train_set本身的值
    # 目的是为了将预测器与标签分开
    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        housing.columns.get_loc(c) for c in col_names]  # get the column indices

    # 那么这个流水线就是对数据依次进行插值，添加组合特征，特征缩放
    # PS：均是对数值特征进行处理
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(rooms_ix, bedrooms_ix, population_ix, households_ix)),
        ('std_scaler', StandardScaler()),
    ])

    # 这一部分对数值特征与非数值特征均可处理
    # 获取数值特征的列名
    # PS：housing_num为dataframe类型
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    # 获取文本属性的列名
    cat_attribs = ["ocean_proximity"]

    # List of (name, transformer, columns) tuples specifying the transformer objects
    # to be applied to subsets of the data.
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)


    if num_pipeline['attribs_adder'].add_bedrooms_per_room:
        additional_labels = ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
    else:
        additional_labels = ['rooms_per_household', 'population_per_household']
    labels = num_attribs + additional_labels + list(full_pipeline.named_transformers_.cat.categories_[0])

    return housing_prepared, housing_labels, labels


def select_and_train_a_model(housing_prepared, housing_labels, text_labels):
    def train_lr():
        """ 线性回归模型"""
        print(f'LinearRegression'.center(100, '*'))
        lin_reg = LinearRegression(n_jobs=-1)  # n_jobs=-1使用全部核数
        lin_reg.fit(housing_prepared, housing_labels)
        # 计算在整个训练集上的RMSE
        housing_predictions = lin_reg.predict(housing_prepared)
        lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
        print(f'no CV version'.center(75, '*'))
        print(f'线性回归损失：{lin_rmse}')
        print(f'线性回归模型斜率：{lin_reg.coef_}')
        print(f'线性回归模型截距：{lin_reg.intercept_}')
        utils.show_feature_coef(lin_reg.coef_, text_labels=text_labels, bottom=0.45,
                                title='Corresponding coefficient size of each feature')

        # 交叉验证的线性回归模型
        # 计算线性模型的交叉验证作为对比
        print(f'CV version'.center(75, '*'))
        lin_scores = cross_val_score(LinearRegression(), housing_prepared, housing_labels,
                                     scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        print(f'交叉验证线性回归损失：')
        utils.display_scores(lin_rmse_scores)

    def train_dtr():
        """决策树回归模型"""
        print(f'DecisionTreeRegressor'.center(100, '*'))
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)
        housing_predictions = tree_reg.predict(housing_prepared)
        tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
        # 误差为0，很可能是严重过拟合了，毕竟用训练集训练与测试，发生了信息泄露，但是这样省时间，
        # 所以也不是完全没有可取之处？
        print(f'no CV version'.center(75, '*'))
        print(f'决策树损失：{tree_rmse}')

        # 使用交叉验证的决策树模型
        scores = cross_val_score(DecisionTreeRegressor(), housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        utils.display_scores(tree_rmse_scores)
        print(f'CV version'.center(75, '*'))
        print(f'交叉验证决策树回归损失：')

    def train_rfr():
        """随机森林回归模型"""
        forest_reg = RandomForestRegressor()
        forest_reg.fit(housing_prepared, housing_labels)
        housing_predictions = forest_reg.predict(housing_prepared)
        forest_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
        print(f'no CV version'.center(75, '*'))
        print(f'随机森林回归损失：{forest_rmse}')

        # 交叉验证的随机森林
        forest_scores = cross_val_score(RandomForestRegressor(), housing_prepared, housing_labels,
                                        scoring="neg_mean_squared_error", cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        print(f'交叉验证随机森林回归损失：')
        utils.display_scores(forest_rmse_scores)

    def train_svr():
        """支持向量机回归模型"""
        svm_reg = SVR(kernel="linear")
        svm_reg.fit(housing_prepared, housing_labels)
        housing_predictions = svm_reg.predict(housing_prepared)
        svm_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
        print(f'no CV version'.center(75, '*'))
        print(f'SVR回归损失：{svm_rmse}')

        # 交叉验证的支持向量机回归
        svr_scores = cross_val_score(SVR(kernel="linear"), housing_prepared, housing_labels,
                                     scoring="neg_mean_squared_error", cv=10)
        svr_rmse_scores = np.sqrt(-svr_scores)
        print(f'交叉验证SVR回归损失：')
        utils.display_scores(svr_rmse_scores)


    # train_lr()
    # train_dtr()
    # train_rfr()
    # train_svr()


def fine_tune_your_model(housing_prepared, housing_labels, text_labels):
    def run_grid_search():
        print('GRID SEARCH'.center(100, '*'))
        # list类型（其中元素为dict类型），记录了要寻找的超参数（n_estimators与max_features）
        # （类RandomForestRegressor传入的变量），
        # 与超参数寻找的范围（不知道超参数应该赋什么值时，可以尝试从10的连续次幂寻找）
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                                   return_train_score=True, verbose=10)
        grid_search.fit(housing_prepared, housing_labels)

        print('返回最佳的超参数组合：')
        print(grid_search.best_params_)
        print('返回采用此最佳超参数组合的分类器：')
        print(grid_search.best_estimator_)

        cvres = grid_search.cv_results_
        # 返回18组超参数搜索的，每组参数对应的平均交叉验证分数（5次），与对应超参数组
        print('返回18组超参数搜索的，每组参数对应的平均交叉验证分数（5次），与对应超参数组：')
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

    def run_randomized_search():
        print('RANDOMIZED SEARCH'.center(100, '*'))
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        # n_iter=10，进行了10次随机搜索
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10, cv=5,
                                        scoring='neg_mean_squared_error', verbose=10)
        rnd_search.fit(housing_prepared, housing_labels)

        print('返回最佳的超参数组合：')
        print(rnd_search.best_params_)
        print('返回采用此最佳超参数组合的分类器：')
        print(rnd_search.best_estimator_)

        print('返回10组超参数搜索的平均交叉验证分数（5次），与对应超参数组：')
        cvres = rnd_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

    # run_grid_search()
    run_randomized_search()



if __name__ == '__main__':
    np.random.seed(42)

    housing = take_a_quick_look_at_the_data_structure()
    strat_train_set = create_a_test_set(housing)
    # discover_and_visualize_the_data_to_gain_insights(strat_train_set)
    housing_prepared, housing_labels, text_labels = prepare_the_data_for_machine_learning_algorithms(strat_train_set)

    # select_and_train_a_model(housing_prepared, housing_labels, text_labels)
    fine_tune_your_model(housing_prepared, housing_labels, text_labels)
