"""
LDA与PCA
"""
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.metrics as sm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
x_data = iris.data
y_data = iris.target
x_data = pd.DataFrame(x_data, columns=iris.feature_names)
y_data = pd.DataFrame(y_data, columns=["features_name"])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 不做任何变换，调用KNN模型，取基线准确率
clf = LogisticRegression()
knn_average = cross_val_score(clf, x_train, y_train).mean()
print("========================================================")
clf = clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
print("训练集交叉验证基线准确率:%.4f" % knn_average)
print("训练集基线准确率:%.4f" % sm.accuracy_score(y_train, pred_train))
print("测试集基线准确率:%.4f" % sm.accuracy_score(y_test, pred_test))
print("训练集混淆矩阵:", "\n", sm.confusion_matrix(y_train, pred_train))
print("测试集混淆矩阵:", "\n", sm.confusion_matrix(y_test, pred_test))


# 加入LDA进行特征转换（LDA降维后，效果提升6%-9%,测试集acc为100%）
single_lda = LinearDiscriminantAnalysis(n_components=2)
lda_pipeline = Pipeline([("lda", single_lda), ("clf", LogisticRegression())])
lda_average = cross_val_score(lda_pipeline, x_train, y_train).mean()

LDA = single_lda.fit(x_train, y_train)
x_train = LDA.transform(x_train)
x_test = LDA.transform(x_test)
clf = LogisticRegression().fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
print("=====================LDA特征转换效果======================")
print("LDA保留最好的线性判别式:%.4f" % lda_average)
print("训练集LDA降维准确率:%.4f" % sm.accuracy_score(y_train, pred_train))
print("测试集LDA降维准确率:%.4f" % sm.accuracy_score(y_test, pred_test))
print("训练集LDA降维混淆矩阵:", "\n", sm.confusion_matrix(y_train, pred_train))
print("测试集LDA降维混淆矩阵:", "\n", sm.confusion_matrix(y_test, pred_test))


# 加入PCA进行特征转换(PCA进行特征转换后，测试集效果下降)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

single_pca = PCA(n_components=2)
pca_pipeline = Pipeline([("pca", single_pca), ("clf", LogisticRegression())])
pca_average = cross_val_score(pca_pipeline, x_train, y_train).mean()

pca = single_pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
clf = LogisticRegression().fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
print("=====================PCA特征转换效果======================")
print("PCA保留最好的线性判别式:%.4f" % lda_average)
print("训练集PCA降维准确率:%.4f" % sm.accuracy_score(y_train, pred_train))
print("测试集PCA降维准确率:%.4f" % sm.accuracy_score(y_test, pred_test))
print("训练集PCA降维混淆矩阵:", "\n", sm.confusion_matrix(y_train, pred_train))
print("测试集PCA降维混淆矩阵:", "\n", sm.confusion_matrix(y_test, pred_test))


# 使用特征选择工具与特征转换做对比
from sklearn.feature_selection import SelectKBest

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)
scaler = MinMaxScaler().fit(x_train)
x_train1 = scaler.transform(x_train)
x_test1 = scaler.transform(x_test)

for k in [1, 2, 3]:
    # 构建流水线
    k_best = SelectKBest(k=k)
    select_pipeline = Pipeline([("swlect", k_best), ("clf", LogisticRegression())])
    # 交叉验证流水线
    select_average = cross_val_score(select_pipeline, x_train1, y_train).mean()

    k_best = k_best.fit(x_train1, y_train)
    x_train = k_best.transform(x_train1)
    x_test = k_best.transform(x_test1)

    clf = LogisticRegression().fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    print("特征选择{}个特征，交叉验证效果:".format(k), np.round(select_average, 4))
    print("特征选择{}个特征，训练集效果:".format(k), np.round(sm.accuracy_score(y_train, pred_train), 4))
    print("特征选择{}个特征，测试集效果:".format(k), np.round(sm.accuracy_score(y_test, pred_test), 4))

"""
分析可知，LDA特征转换效果最好，其次是使用SelectKBest进行特征选择，PCA效果并不理想，很大可能是由于数据维度太低
"""



"""
建立大型流水线，一次性搜索LDA，PCA，缩放，SelectKBest最佳参数
"""
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

def get_best_model_and_accuracy(model, params, x, y):
    grid = GridSearchCV(model,             # 网格搜索的模型
                        params,            # 实验的参数
                        error_score=0.     # 如果出错，当做结果为0
                        )
    grid.fit(x, y)
    # 传统的性能指标
    print("Best Accuracy: {}".format(grid.best_score_))
    # 最好参数
    print("Best Parameters: {}".format(grid.best_params_))
    # 平均拟合时间
    print("Average Time to Fit (s): {}".format(round(grid.cv_results_["mean_fit_time"].mean(), 3)))
    # 平均预测时间(秒)
    # 显示模型在实时分析中的性能
    print("Average Time to Score (s): {}".format(round(grid.cv_results_["mean_score_time"].mean(), 3)))

iris_params = {
                "preprocessing__scale__with_std": [True, False],
                "preprocessing__scale__with_mean": [True, False],
                "preprocessing__pca__n_components": [1, 2, 3, 4],
                # 根据sklearn文档，LDA最大n_components是类别数减1
                "preprocessing__lda__n_components": [1, 2],
                "clf__penalty": ["l1", "l2"],
                "clf__C": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1]
                }
# 更大的流水线
preprocessing = Pipeline([("scale", StandardScaler()),
                          ("pca", PCA()),
                          ("lda", LinearDiscriminantAnalysis())])
iris_pipeline = Pipeline(steps=[("preprocessing", preprocessing),
                                ("clf", LogisticRegression())])
get_best_model_and_accuracy(iris_pipeline, iris_params, x=x_train, y=y_train)

# 通过流水线后，对数据进行拟合与预测（这里没有进行网格搜索，类似于直接对模型拟合预测，只是对其他过程进行了流水线处理）
iris_pipeline.fit(x_train, y_train)
print("训练集acc:%.4f" % (iris_pipeline.score(x_train, y_train)))
print("测试集acc:%.4f" % (iris_pipeline.score(x_test, y_test)))