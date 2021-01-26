import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sm
from sklearn.pipeline import Pipeline
import seaborn as sb
from scipy import stats
plt.style.use("ggplot")

data = pd.read_csv("activity_recognizer.csv", header=None)
data.columns = ["index", "x", "y", "z", "activity"]

# 数据探索
# print(data.info())
# print(data.isnull().sum())

x_data = data[["x", "y", "z"]]
y_data = data["activity"]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

# 查看空准确率
score_raw = data["activity"].value_counts(normalize=True)
# print("空准确率为:", "\n",  score_raw)

# 网格搜索，寻找最佳参数
# knn_params = {"n_neighbors": [i for i in range(60)]}
# clf = KNeighborsClassifier()
# grid = GridSearchCV(clf, knn_params)
# grid.fit(x_train, y_train)
# print("最佳参数:", grid.best_params_, "最佳得分:", grid.best_score_)

# 预测
clf = KNeighborsClassifier(n_neighbors=31).fit(x_train, y_train)
prd_train = clf.predict(x_train)
prd_test = clf.predict(x_test)
print("===============原始数据===================")
print("训练集acc:%.4f" % (sm.accuracy_score(y_train, prd_train)))
print("测试集acc:%.4f" % (sm.accuracy_score(y_test, prd_test)))
# print("训练集混淆矩阵:", "\n", sm.confusion_matrix(y_train, prd_train))
# print("测试集混淆矩阵:", "\n", sm.confusion_matrix(y_test, prd_test))


# 特征构造（多项式特征）
from sklearn.preprocessing import PolynomialFeatures
print("===============多项式特征构造===================")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
x_poly = poly.fit_transform(x_data)
x_poly = pd.DataFrame(x_poly, columns=poly.get_feature_names())
# print(x_poly)

# corr = abs(x_poly.corr())
# sb.heatmap(corr, vmax=1, vmin=0, annot=True)
# plt.show()

# 设置流水线参数，实例化流水线，使用网格搜索，得到最佳参数
# clf = KNeighborsClassifier()
# pipe_params = {"poly_features__degree": [1, 2, 3], "poly_features__interaction_only": [True, False], "classify__n_neighbors": [i for i in range(1, 60)]}
# pipe = Pipeline([("poly_features", poly), ("classify", clf)])
# grid = GridSearchCV(pipe, pipe_params)
# grid.fit(x_train, y_train)
# print("最佳参数:", grid.best_params_, "最佳得分:", grid.best_score_)

clf = KNeighborsClassifier(n_neighbors=41)
x_poly1 = PolynomialFeatures(degree=3, interaction_only=False).fit_transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_poly1, y_data, train_size=0.7, random_state=66)

clf.fit(x_train, y_train)
prd_train = clf.predict(x_train)
prd_test = clf.predict(x_test)
print("训练集acc:%.4f" % (sm.accuracy_score(y_train, prd_train)))
print("测试集acc:%.4f" % (sm.accuracy_score(y_test, prd_test)))
# print("训练集混淆矩阵:", "\n", sm.confusion_matrix(y_train, prd_train))
# print("测试集混淆矩阵:", "\n", sm.confusion_matrix(y_test, prd_test))