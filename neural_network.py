# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     neural_network
   Description :
   Author :        chen
   date：          2019/6/29
-------------------------------------------------
   Change Activity:
                   2019/6/29:
-------------------------------------------------
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

FILE_PATH = './dataset/sample_big.csv'

#定义sigmod函数及其导数
def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

#定义NeuralNetwork 神经网络算法
class NeuralNetwork:
    def __init__(self,neuron_list):
        """
        初始化神经网络类，

        example:
        eg[2,2,1]表示第一层2个神经元，第二层2个神经元，第三层1个神经元

        :param neuron_list: 各层的神经元
        """

        self.activation = logistic
        self.activation_deriv = logistic_derivative

        self.weights = []

        #循环从1开始，相当于以第二层为基准，进行权重的初始化
        for i in range(1, len(neuron_list) - 1):
            #对当前神经节点的前驱赋值
            self.weights.append((2*np.random.random((neuron_list[i - 1] + 1, neuron_list[i] + 1))-1)*0.25)
            #对当前神经节点的后继赋值
            self.weights.append((2*np.random.random((neuron_list[i] + 1, neuron_list[i + 1]))-1)*0.25)


    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        """
        训练函数

        example:
        X_train = [ [] [] ... [] ]
        labels_train = [ 1, 2, 3, ... , 10 ] or [ [1,0,0] [0,1,0] [0,0,1] ]
        nn.fit(X_train,labels_train)

        :param X: 矩阵，训练集
        :param y: 训练集对应的标签集，多标签则用二维数组表示
        :param learning_rate: 学习率
        :param epochs: 表示抽样的方法对神经网络进行更新的最大次数
        :return:
        """
        X = np.atleast_2d(X) #确定X至少是二维的数据
        temp = np.ones([X.shape[0], X.shape[1]+1]) #初始化矩阵
        # print(temp)
        temp[:, 0:-1] = X  # adding the bias unit to the input layer

        X = temp
        # print(X)
        y = np.array(y) #把list转换成array的形式

        for k in range(epochs):
            #随机选取一行，对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            #完成所有正向的更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            #开始反向计算误差，更新权重
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict_one(self, x):
        """
        预测函数
        :param x:  test_x 集合
        :return:  预测结果 [ , , ... ,  ]
        """
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def predict_all(self,test_x,test_y):
        """
        一次性预测多个数据
        :param test_x: 测试集的集合
        :return:混淆矩阵和分类情况
        """
        count = 0
        predictions = []
        for index, row_x in test_x.iterrows():
            o = self.predict_one(row_x)
            res = np.argmax(o)
            predictions.append(res)
            # name = get_key(namedict,res)
            # print(name)
        confusion = confusion_matrix(test_y, predictions)
        class_report = classification_report(test_y, predictions)
        return confusion,class_report


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def fit_transform(data_set,cls_name):
    """
    转化标签类为数字，并保存这个key value
    :param data_set:
    :param cls_name:
    :return:
    """
    acol = set(data_set[cls_name].tolist())
    dict = {}
    index = 0
    for i in acol:
        dict[i]=index
        index += 1

    return dict,data_set[cls_name].map(dict)