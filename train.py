# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:10:00 2022
训练，保存标准化数据scaler/模型
@author: 99773

"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
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

data_x = fluxs
data_y = labels

scaler = preprocessing.StandardScaler().fit(data_x)
data_x = scaler.transform(data_x)
# data_train, data_test, data_train_label, data_test_label = train_test_split(data_x,data_y, test_size=0.3, random_state=32)
data_train, data_test, data_train_label, data_test_label = \
    train_test_split(data_x,data_y, test_size=0.1, random_state=32, stratify= data_y)

print('data_tain',data_train.shape, data_train_label.shape)
print('data_test',data_test.shape, data_test_label.shape)

"""搭建网络"""
tot = len(data_x[0,:])
model_mlp = MLPClassifier(
    # hidden_layer_sizes = (tot, 2000, 1000, 500, 500, 100, 50, 3),
    hidden_layer_sizes = (1000, 500, 250, 100, 50),
    activation = 'relu',
    solver = 'adam',
    max_iter = 200,
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

# =============================================================================
# # 加载网络，用于复现训练结果
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
# =============================================================================
# # 相关性
# plt.figure('Train and Test')
# plt.subplot(121),plt.plot(data_train_label, p_train,'.g')
# # 误差均方差
# err_train = (data_train_label-p_train).std()
# # 相关系数
# s1 = np.polyfit(data_train_label, p_train, 1)
# p1 = np.poly1d(s1)
# pr1 = pr(data_train_label, p_train)
# plt.title('Trainset Err:'+str(err_train)[:5]+' cc: '+str(pr1)[1:6])
# plt.xlabel('real')
# plt.ylabel('predict')
# plt.subplot(122)
# plt.plot(data_test_label, p_test,'.r')
# err_test = (data_test_label-p_test).std()
# # 相关系数
# s1_test = np.polyfit(data_test_label, p_test,1)
# p1_test = np.poly1d(s1_test)
# pr1_test = pr(data_test_label, p_test)
# plt.title('Testset Err:'+str(err_test)[:5]+' cc: '+str(pr1_test)[1:6])
# plt.xlabel('real')
# plt.ylabel('predict')
# 
# plt.suptitle('nn_patch')
# # res
# plt.figure('Res_train'),plt.plot(data_train_label-p_train,'.'),plt.title('res_train')
# plt.figure('Res_test'),plt.plot(data_test_label-p_test,'.'),plt.title('res_test')
# =============================================================================

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




