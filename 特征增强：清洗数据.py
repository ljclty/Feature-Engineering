"""
匹马印第安人糖尿病预测数据集
pima.data
"""

# 1、导库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
# plt.style.use("fivethirtyeight")

# 2、数据读取(原始数据无列名，手动添加)
data = pd.read_csv("pima.data", header=None)
data.columns = ["times_pregnant", "plasma_glucose_concentration", "diastolic_blood_pressure", "triceps_thickness", "serum_insulin", "bmi", "pedigree_function", "age", "onset_diabetes"]

# 3、数据探索性分析（有无缺失值，每一列数据类型，基本概率统计）
# print(data.info())
pd.set_option('display.max_columns', 1000)      # 可显示所有数据，无省略号
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
raw_data = data.describe()
# print("数据填充之前:", data.describe())

# 4、对异常数据进行预处理,删除或者填充（数据探索性分析中，异常值，与数据理解差别很大的数据,视为缺失值）
columns = ["plasma_glucose_concentration", "diastolic_blood_pressure", "triceps_thickness", "serum_insulin", "bmi"]
for column in columns:
    data[column].replace([0], np.nan, inplace=True)
data.dropna(inplace=True)
# print(data.describe())
# print(data.isnull().sum())

# 5、数据初步处理完成，进行变量相关性分析（可视化与相关性矩阵）
# sns.heatmap(data.corr())
# plt.show()
# print(data.corr()["onset_diabetes"].sort_values(ascending=False))

# 6、找出最有影响点量，再可视化
"""
for col in ["bmi", "diastolic_blood_pressure", "plasma_glucose_concentration"]:
    plt.hist(data[data["onset_diabetes"] == 0][col], bins=10, alpha=0.5, label="non-diabetes")
    plt.hist(data[data["onset_diabetes"] == 1][col], bins=10, alpha=0.5, label="diabetes")
    plt.legend(loc="upper right")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title("Histogram of {}".format(col))
    plt.show()
"""

# 7、检查数据缺失值填充后，数据整体的分布改变情况
"""
process_data = data.describe()
# print("数据填充之后:", data.describe())
# 使用条形图对变化可视化
change = (process_data.loc["mean", :] - raw_data.loc["mean", :]) / (raw_data.loc["mean", :])
# print(change)
ax = change.plot(kind="bar", title="% change in average column values")
ax.set_ylabel("% change")
plt.show()
"""

# 8、开始机器学习
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as sm

x_data = data.iloc[:, : -1]
y_data = data["onset_diabetes"]
x_trian, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

knn_params = {"n_neighbors":  [i for i in range(1, 30)]}
clf = KNeighborsClassifier()
grid = GridSearchCV(clf, knn_params)
grid.fit(x_trian, y_train)
print("最佳参数:", grid.best_params_, "最佳得分:", grid.best_score_)
clf = KNeighborsClassifier(n_neighbors=9).fit(x_trian, y_train)
prd_test = clf.predict(x_test)
prd_train = clf.predict(x_trian)
print("================删除缺失值================")
print("训练集精度:%.2f" % (sm.accuracy_score(y_train, prd_train) * 100), "%")
print("测试集精度:%.2f" % (sm.accuracy_score(y_test, prd_test) * 100), "%")
print("训练集混淆矩阵:", "\n", sm.confusion_matrix(y_train, prd_train))
print("测试集混淆矩阵:", "\n", sm.confusion_matrix(y_test, prd_test))


# 使用sklearn进行缺失值填充
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
data = pd.read_csv("pima.data", header=None)
data.columns = ["times_pregnant", "plasma_glucose_concentration", "diastolic_blood_pressure", "triceps_thickness", "serum_insulin", "bmi", "pedigree_function", "age", "onset_diabetes"]
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
x_data = data.iloc[:, : -1]
y_data = data["onset_diabetes"]
x_trian, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

knn_params = {"n_neighbors":  [i for i in range(1, 30)]}
clf = KNeighborsClassifier()
grid = GridSearchCV(clf, knn_params)
grid.fit(x_trian, y_train)
print("最佳参数:", grid.best_params_, "最佳得分:", grid.best_score_)
clf = KNeighborsClassifier(n_neighbors=21).fit(x_trian, y_train)
prd_test = clf.predict(x_test)
prd_train = clf.predict(x_trian)
print("================错误填充缺失值方法================")
print("训练集精度:%.2f" % (sm.accuracy_score(y_train, prd_train) * 100), "%")
print("测试集精度:%.2f" % (sm.accuracy_score(y_test, prd_test) * 100), "%")
print("训练集混淆矩阵:", "\n", sm.confusion_matrix(y_train, prd_train))
print("测试集混淆矩阵:", "\n", sm.confusion_matrix(y_test, prd_test))

# 填充缺失值不能一次性对整个数据集进行填充，应该先对训练集进行，然后按照训练集的信息，对测试集填充
from sklearn.pipeline import Pipeline

data = pd.read_csv("pima.data", header=None)
data.columns = ["times_pregnant", "plasma_glucose_concentration", "diastolic_blood_pressure", "triceps_thickness", "serum_insulin", "bmi", "pedigree_function", "age", "onset_diabetes"]
x_data = data.iloc[:, : -1]
y_data = data["onset_diabetes"]
x_trian, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

knn_params = {"classify__n_neighbors":  [i for i in range(1, 30)]}
clf = KNeighborsClassifier()
median_impute = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("classify", clf)])
grid = GridSearchCV(median_impute, knn_params)
grid.fit(x_trian, y_train)
print("最佳参数:", grid.best_params_, "最佳得分:", grid.best_score_)
clf = KNeighborsClassifier(n_neighbors=21).fit(x_trian, y_train)
prd_test = clf.predict(x_test)
prd_train = clf.predict(x_trian)
print("================正确填充缺失值方法================")
print("训练集精度:%.2f" % (sm.accuracy_score(y_train, prd_train) * 100), "%")
print("测试集精度:%.2f" % (sm.accuracy_score(y_test, prd_test) * 100), "%")
print("训练集混淆矩阵:", "\n", sm.confusion_matrix(y_train, prd_train))
print("测试集混淆矩阵:", "\n", sm.confusion_matrix(y_test, prd_test))


