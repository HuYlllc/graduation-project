import numpy as np
import pandas as pd
import csv
from sklearn.cluster import Birch
import math
from collections import Counter
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
global label_list  # label_list为全局变量

# 将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x, y):
    return [i for i in range(len(y)) if y[i] == x]

def handleLabel(input):
    label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    'spy.', 'rootkit.']
    # global label_list  # 在函数内部使用全局变量并修改它
    if input[41] in label_list:
        return find_index(input[41], label_list)[0]
    else:
        label_list.append(input[41])
        return find_index(input[41], label_list)[0]

def one_hot(): #热编码
    source_target = []
    source_file = 'kddcup.data_10_percent_corrected'
    handled_file= 'kddcup.data_10_percent_corrected.csv'
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        for row in csv_reader:
            source_target.append(handleLabel(row))
        source_target = to_categorical(source_target)
    # print(source_target[0])
    np_data = [np.argmax(one_hot) for one_hot in source_target]
    # print(np_data)
    for i in range(len(np_data)):
        if np_data[i] != 0:
            np_data[i] = 1
    return np_data

def change(x):
    if x!= 0:
        x = 1
    return x

def getdata(dataname = u"kddcup.data_10_percent_corrected.csv"):
    data_label = pd.read_csv(dataname, encoding="utf-8", header=None, nrows=40000)
    data_label[41] = data_label[41].map(lambda x: change(x))
    dataset = data_label
    return dataset

def birch(data):
    X = data
    birch = Birch(n_clusters=2, threshold=0.5)
    ##训练数据
    labels = birch.fit_predict(X)
    print(Counter(labels))
    return labels



def ent(array, values, num_label):
    count = 0
    for i in array:
        if i == values:
            count += 1
    temp_ent = -(count / num_label) * math.log(count / num_label)
    return temp_ent

# def feature_choose(data, choose_array): # 输入特征dataframe和不需要的特征，得到选择后的特征
#     for i in choose_array:
#         data.pop(i)
#     return data

class InformationGain():
    def __init__(self, feature, label):
        feature = np.array(feature)
        num_of_label = len(label)
        temp_ent = 0
        shanno_ent = []
        # Counter(label)
        temp_ent = ent(label, 0, num_of_label) + ent(label, 1, num_of_label)
        shanno_ent.append(temp_ent)
        self.shannoEnt = shanno_ent[0]
    def getEnt(self):
        return self.shannoEnt

## 生成训练集和测试集
def trainAndtest(data_X, data_y, batch_size):
    data_X.pop(41)
    data_X = np.array(data_X)
    data_X = np.pad(data_X, ((0, 0), (0, 64 - len(data_X[0]))), 'constant').reshape(-1, 1, 8, 8)
    data_y = np.array(data_y)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
    train_dataset = TensorDataset(
    torch.from_numpy(X_train.astype(np.float32)),
    torch.from_numpy(y_train.astype(np.int64))
    )
    test_dataset = TensorDataset(
    torch.from_numpy(X_test.astype(np.float32)),
    torch.from_numpy(y_test.astype(np.int64))
    )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_dataloader, test_dataloader, len(X_train), len(X_test)


if __name__ == '__main__':
    label = one_hot()
    data = getdata()
    # print(data)
    # birch(data)
    # # print(type(birch(data)), type(label))
    # shannoEntarray = []
    # for i in range(0, data.shape[1]-1):
    #     print(i)
    #     # print(data.iloc[0:20000, i].tolist())
    #     shannoEntarray.append(InformationGain(data, birch(data.iloc[0:20000, [i, 41]])).getEnt())
    # print(shannoEntarray)
    # dataset = trainAndtest(data, data[41])
    # print(InformationGain(data, birch(data)).getEnt())
