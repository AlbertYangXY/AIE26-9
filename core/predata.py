# -*- coding:utf-8 -*-
'''
首先对数据进行截取，获得id 、answer和是非观点极性
'''
import os 
import numpy as np
import json
import jieba
import pickle
import codecs
# 设置环境变量
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trainpath = os.path.join(BaseDir,"data/train.json")
testpath = os.path.join(BaseDir,"data/test1.json")
devpath = os.path.join(BaseDir,"data/dev.json")
pklpath = os.path.join(BaseDir,"data/data.pkl")

# 读取文件，获取answer和是否极性
with open(trainpath,"r",encoding="utf-8") as f:
    files_data = f.readlines()
    x_train_id = [ json.loads(itr)["id"] for itr in files_data]
    # x_train_id = np.array[x_train_id]
    x_train_answer = [ json.loads(itr)["answer"] for itr in files_data]
    x_train_question = [json.loads(itr)["question"] for itr in files_data]
    # x_train_answer = np.array[x_train_answer]
    y_train_label = [ json.loads(itr)["yesno_answer"] for itr in files_data]
    # y_train_label = np.array[y_train_label]

with open(devpath,"r",encoding="utf-8") as f:
    files_data = f.readlines()
    x_dev_id = [ json.loads(itr)["id"] for itr in files_data]
    # x_dev_id = np.array[x_dev_id]
    x_dev_answer = [ json.loads(itr)["answer"] for itr in files_data]
    x_dev_question = [json.loads(itr)["question"] for itr in files_data]
    # x_dev_answer = np.array[x_dev_answer]
    y_dev_label = [ json.loads(itr)["yesno_answer"] for itr in files_data]
    # y_dev_label = np.array[y_dev_label]

with open(testpath,"r",encoding="utf-8") as f:
    files_data = f.readlines()
    x_test_id = [ json.loads(itr)["id"] for itr in files_data]
    # x_test_id = np.array[x_test_id]
    x_test_answer = [ json.loads(itr)["answer"] for itr in files_data]
    x_test_question = [json.loads(itr)["question"] for itr in files_data]
    # x_test_answer = np.array[x_test_answer]

def get_words(lanswer):
    words = []
    for itr in lanswer:
        seg_list = jieba.cut(itr)
        words.append(" ".join(seg_list))
    return words
label2int = {"Yes": 1,"No":2,"Depends":3}
int2label = {1:"Yes",2:"No",3:"Depends"}
y_train_label = [label2int[itr] for itr in y_train_label]
x_train_answer = get_words(x_train_answer)
x_train_question = get_words(x_train_question)
y_dev_label = [label2int[itr] for itr in y_dev_label]
x_dev_answer = get_words(x_dev_answer)
x_dev_question = get_words(x_dev_question)
x_test_answer = get_words(x_test_answer)
x_test_question = get_words(x_test_question)


with open(pklpath,"wb") as f:
    pickle.dump((x_train_id,x_train_question,x_train_answer,y_train_label),f)
    pickle.dump((x_dev_id, x_dev_question, x_dev_answer, y_dev_label), f)
    pickle.dump((x_test_id, x_test_question, x_test_answer), f)

# with open(pklpath,"rb") as f:
#     x_train_id, x_train_question, x_train_answer, y_train_label = pickle.load(f)
# print(x_train_id[:10])
# print(x_train_question[:10])
# print(x_train_answer[:10])
# print(y_train_label[:10])


# np.savez("data.npz",x_train_id=)

