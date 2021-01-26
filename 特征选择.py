# 1.导库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# 2.划分数据集
data = pd.read_csv("credit_card_default.csv")
x_data = data.iloc[:, : -1]
y_data = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=66)
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 3.计算空准确率
print("空准确率:", y_data.value_counts(normalize=True).values[0])


# 4.实例化模型，并且设置参数
lr = LogisticRegression()
knn = KNeighborsClassifier()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
lr_params = {"C": [1e-1, 1e0, 1e1, 1e2], "penalty": ["l1", "l2"]}
knn_params = {"n_neighbors": [i for i in range(50)]}
tree_params = {"max_depth": [None, [i for i in range(30)]]}
forest_params = {"n_estimators": [10, 40, 80, 100, 120, 140, 160, 180, 200], "max_depth": [None, [i for i in range(30)]]}

# 5.导入网络搜索模块
from sklearn.model_selection import GridSearchCV

def get_best_model_and_accuracy(model, params, x, y):
    grid = GridSearchCV(model,   # 要搜索的模型
                        params,  # 要尝试的参数
                        error_score=0.  # 如果报错，结果是0
                        )
    grid.fit(x, y)     # 拟合模型和参数
    print("最佳得分:{}".format(grid.best_score_))  # 模型一个指标
    print("最佳得分参数:{}".format(grid.best_params_))  # 模型参数
    print("模型拟合时间:{}".format(round(grid.cv_results_["mean_fit_time"].mean(), 3)))
    print("模型预测时间:{}".format(round(grid.cv_results_["mean_score_time"].mean(), 3)))
"""
# 开始查看模型性能
print("=================LogisticRegression=======================")
get_best_model_and_accuracy(lr, lr_params)
print("=================KNeighborsClassifier=======================")
get_best_model_and_accuracy(knn, knn_params)
print("=================DecisionTreeClassifier=======================")
get_best_model_and_accuracy(tree, tree_params)
print("=================RandomForestClassifier=======================")
get_best_model_and_accuracy(forest, forest_params)
"""



"""
基于相关性进行特征选择
"""
# 热力图(自动选择最相关的特征)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
sns.heatmap(data.corr())
plt.show()

# print("特征与响应变量的相关性:", "\n", abs(data.corr()["default payment next month"]).sort_values(ascending=False))
# print("相关性大于0.2的特征:", "\n", data.columns[abs(data.corr()["default payment next month"]) > 0.2])
high_correlated_features = data.columns[abs(data.corr()["default payment next month"]) > 0.2].drop("default payment next month")
x_data = data[high_correlated_features]
y_data = data["default payment next month"]
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, train_size=0.8, random_state=66)

# get_best_model_and_accuracy(tree, tree_params, x=x_train1, y=y_train1)
# get_best_model_and_accuracy(lr, lr_params, x=x_train1, y=y_train1)
# get_best_model_and_accuracy(knn, knn_params, x=x_train1, y=y_train1)
# get_best_model_and_accuracy(forest, forest_params, x=x_train1, y=y_train1)




""""
给予假设检验，进行特征选择
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

k_best = SelectKBest(f_classif, k=5)
k_best = k_best.fit(x_train1, y_train1)
x_train2 = k_best.transform(x_train1)
x_test2 = k_best.transform(x_test1)

p_values = pd.DataFrame({"column": x_train1.columns, "p_value": k_best.pvalues_}).sort_values("p_value")
# print(p_values)

# get_best_model_and_accuracy(tree, tree_params, x=x_train2, y=y_train1)
# get_best_model_and_accuracy(lr, lr_params, x=x_train1, y=y_train1)
# get_best_model_and_accuracy(knn, knn_params, x=x_train1, y=y_train1)
# get_best_model_and_accuracy(forest, forest_params, x=x_train1, y=y_train1)



"""
基于模型的特征选择
"""
from sklearn.feature_selection import SelectFromModel
select_from_model = SelectFromModel(DecisionTreeClassifier(), threshold=0.05)
select_ = select_from_model.fit(x_train, y_train)
select_x_train = select_.transform(x_train)
select_x_test = select_.transform(x_test)
# get_best_model_and_accuracy(tree, tree_params, x=select_x_train, y=y_train1)



"""
使用逻辑回归作为选择特征，用决策树进行评估，使用网格搜索
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
logistic_selector = SelectFromModel(LogisticRegression())
regularization_pipe = Pipeline([("select", logistic_selector), ("classifier", tree)])
regularization_pipe_params = {"classifier__max_depth": [i for i in range(30)]}
regularization_pipe_params.update({"select__threshold": [0.01, 0.05, 0.1, "mean", "median", "2.*mean"], "select__estimator__penalty": ["l1", "l2"],})
# get_best_model_and_accuracy(regularization_pipe, regularization_pipe_params, x=x_train, y=y_train)


