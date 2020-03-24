# -*- coding:utf-8 -*-
'''
首先对数据进行截取，获得id 、answer和是非观点极性
'''
import os
import numpy as np
import time
import json
import jieba
import pickle
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
# 设置环境变量
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trainpath = os.path.join(BaseDir,"data/train.json")
testpath = os.path.join(BaseDir,"data/test1.json")
devpath = os.path.join(BaseDir,"data/dev.json")
pklpath = os.path.join(BaseDir,"data/data.pkl")

with open(pklpath,"rb") as f:
    x_train_id, x_train_question, x_train_answer, y_train_label = pickle.load(f)
    x_dev_id, x_dev_question, x_dev_answer, y_dev_label = pickle.load(f)
    x_test_id, x_test_question, x_test_answer = pickle.load(f)

method = CountVectorizer(max_features=20000)
data_vect = method.fit_transform(x_train_answer)
tool_chain = []
tool_chain.append(method)
# method = LatentDirichletAllocation(n_components=30, max_iter=5,
#                                 learning_method='online',
#                                 learning_offset=50.,
#                                 random_state=0)
# st = time.process_time()
# data_dm = method.fit_transform(data_vect)
# ed = time.process_time()
# print("LDA time:", ed-st)
# tool_chain.append(method)

# 随机森林算法
# method = RandomForestClassifier()
# st = time.process_time()
# method.fit(data_vect, y_train_label)
# ed = time.process_time()
# print("RFC time:", ed-st)
# y = method.predict(data_vect)
# print("RFC acc:", np.sum(y==y_train_label)/len(y_train_label))
# tool_chain.append(method)

# 决策树算法
method = DecisionTreeClassifier(random_state=0)
st = time.process_time()
method.fit(data_vect, y_train_label)
ed = time.process_time()
print("RFC time:", ed-st)
y = method.score(data_vect,y_train_label)
print("RFC acc:", y)
tool_chain.append(method)

temp = tool_chain[0].transform(x_dev_answer)
# 使用相同参数进行降维
# temp = tool_chain[1].transform(temp)
# 使用训练好模型进行预测
temp = tool_chain[1].predict(temp)

print("RFC acc:", np.sum(temp==y_dev_label)/len(temp))


