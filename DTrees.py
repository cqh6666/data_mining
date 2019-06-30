# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     DTrees
   Description :
   Author :        chen
   date：          2019/6/29
-------------------------------------------------
   Change Activity:
                   2019/6/29:
-------------------------------------------------
"""

import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_process import get_column_x,get_column_y
#
#
# FILE_PATH = './dataset/sample_big.csv'
# TEST_PATH = './dataset/sample.csv'
#
# train_set = pd.read_csv(FILE_PATH, parse_dates=['Dates'])
# train_set = train_set.drop(['Descript', 'Resolution', 'X', 'Y', 'Address'], axis=1)
# train_set['Dates'] = train_set['Dates'].dt.hour
#
# test_set = pd.read_csv(TEST_PATH, parse_dates=['Dates'])
# test_set = test_set.drop(['Descript', 'Resolution', 'X', 'Y', 'Address'], axis=1)
# test_set['Dates'] = test_set['Dates'].dt.hour
#
#
# Label_type = preprocessing.LabelEncoder()
# crime_type = Label_type.fit_transform(test_set.Category)
# Dates = Label_type.fit_transform(test_set.Dates)
# DayOfWeek = Label_type.fit_transform(test_set.DayOfWeek)
# dis = Label_type.fit_transform(test_set.PdDistrict)
# test_set['Dates'] = Dates
# test_set['DayOfWeek'] = DayOfWeek
# test_set['Category'] = crime_type
# test_set['PdDistrict'] = dis
#
# crime_type = Label_type.fit_transform(train_set.Category)
# Dates = Label_type.fit_transform(train_set.Dates)
# DayOfWeek = Label_type.fit_transform(train_set.DayOfWeek)
# dis = Label_type.fit_transform(train_set.PdDistrict)
# train_set['Dates'] = Dates
# train_set['DayOfWeek'] = DayOfWeek
# train_set['Category'] = crime_type
# train_set['PdDistrict'] = dis
#
# print(train_set,test_set)
#
# column_x = ['Dates','DayOfWeek','PdDistrict']
# column_y = ['Category']
# train_x = train_set[column_x]
# test_x = test_set[column_x]
# test_y = test_set[column_y]
# train_y = train_set[column_y]
# model = tree.DecisionTreeClassifier()
# model.fit(train_x,train_y)
# predicted = model.predict(test_x)
# print(predicted)
# print("决策树准确度:", accuracy_score(x, predicted))


class DTrees:
    """决策树"""
    def __init__(self,train_set,test_set,column_x,column_y):
        """
        决策树构造函数
        :param train_set: 训练集
        :param test_set: 测试集
        :param column_x: 条件属性列表
        :param column_y: 标签属性
        """
        self.train_set = train_set
        self.test_set = test_set
        self.column_x = column_x
        self.column_y = column_y

    def get_train_x_y(self):
        train_x = self.train_set[self.column_x]
        train_y = self.train_set[self.column_y]
        return train_x, train_y

    def get_test_x_y(self):
        test_x = self.test_set[self.column_x]
        test_y = self.test_set[self.column_y]
        return test_x,test_y

