# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:28:20 2024

@author: 99773
"""


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np
import pickle
import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score

# sklearn加速
from sklearnex import patch_sklearn
patch_sklearn()

# 计时开始
start = datetime.datetime.now()

"""数据"""
hdulist = fits.open('train_data_10.fits')
fluxs = hdulist[0].data[::]
objids = hdulist[1].data['objid'][::]
labels = hdulist[1].data['label'][::]

# 计算每个类别的样本数量
class_counts = np.bincount(labels)
# 确定最少的样本数量
min_count = class_counts.min()
# 对每个类别进行欠采样
X_temp, y_temp = fluxs, labels
for class_label in np.unique(labels):
    class_indices = np.where(y_temp == class_label)[0]
    # 随机选择少量样本以匹配最少的样本数量
    if len(class_indices) > min_count:
        # 随机选择索引以匹配最少的样本数量
        np.random.seed(32)
        sampled_indices = np.random.choice(class_indices, min_count, replace=False)
        # 保留随机选择的样本索引，从而删除剩余的样本
        # 使用 np.setdiff1d 来找到不在 sampled_indices 中的索引
        not_sampled_indices = np.setdiff1d(class_indices, sampled_indices)
        # 更新训练集数据，删除未被随机选择的样本
        X_temp = np.delete(X_temp, not_sampled_indices, axis=0)
        y_temp = np.delete(y_temp, not_sampled_indices, axis=0)

# 训练集，测试集
data_x = X_temp
data_y = y_temp
class_count = np.bincount(data_y)   
print('number of classes', class_count) 
scaler = preprocessing.StandardScaler().fit(data_x)
data_x = scaler.transform(data_x)

data_train, data_test, data_train_label, data_test_label = \
    train_test_split(data_x,data_y, test_size=0.1, random_state=32, stratify= data_y)


print('data_tain',data_train.shape, data_train_label.shape)
print('data_test',data_test.shape, data_test_label.shape)

"""搭建网络"""
tot = len(data_x[0,:])
model_mlp = MLPClassifier(

    hidden_layer_sizes = \
        # (5000, 5000, 3000, 2000, 2000, 1000, 500, 100, 50),
        (1000, 500, 250, 100, 50),
    activation = 'relu',
    solver = 'adam',
    max_iter = 300,
    tol = 0.000001,
    random_state = 123,
    verbose = True,
    warm_start = False,
    momentum = 0.9,
    learning_rate = 'adaptive',
    learning_rate_init = 0.001,
    early_stopping = True,
    validation_fraction = 0.15)

# 开始训练
model_mlp.fit(data_train, data_train_label)
train_score = model_mlp.score(data_train,data_train_label)
test_score = model_mlp.score(data_test, data_test_label)

print('训练完成！')
print('train score: ',train_score)
print('test score: ',test_score)

# 加载网络，用于复现训练结果，使用的时候，注释掉搭建的网络就可以了
# =============================================================================
# import joblib
# model_mlp = joblib.load(r'model.m')  #加载模型
# =============================================================================

""""预测"""
p_train = model_mlp.predict(data_train)
p_test = model_mlp.predict(data_test)

accuracy_train = accuracy_score(data_train_label, p_train)
accuracy_test = accuracy_score(data_test_label, p_test)
print('accuracy_train: ',accuracy_train)
print('accuracy_test: ',accuracy_test)

recall_train = recall_score(data_train_label, p_train, average=None)
recall_test = recall_score(data_test_label, p_test, average=None)
print('recall_train: ',recall_train)
print('recall_test: ',recall_test)

precision_train = precision_score(data_train_label, p_train, average=None)
precision_test = precision_score(data_test_label, p_test, average=None)
print('precision_train: ',precision_train)
print('precision_test: ',precision_test)

# create the confusion matrix
confusion_matrix = np.zeros((3,3))
for counter, i in enumerate(data_train_label):
    confusion_matrix[i, p_train[counter]] += 1
from hux import galaxy3_confusion
galaxy3_confusion(confusion_matrix)

"""保存数据"""
# with open('train_test.txt','w') as f:
#     f.write(data_train_label)
#     f.write(p_train)
#     f.write(data_test_label)
#     f.write(p_test)
# f.close()

"""保存模型"""
import joblib
# 保存模型
joblib.dump(model_mlp, r'model.m')
# 保存scaler 
pickle.dump(scaler, open(r'scaler.dump', 'wb'))

# 结束计时
end = datetime.datetime.now()
print('程序运行结束，耗时：', end-start)




