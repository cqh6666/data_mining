# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_process
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
from sklearn.preprocessing import LabelBinarizer


def read_data(file_path):
    """获取csv文件的所有数据作为数据集"""
    data_set = pd.read_csv(file_path,parse_dates = ['Dates'])
    # print(data.keys())
    return data_set

def filter_data(data_set):
    data_set = data_set[['Dates','DayOfWeek','PdDistrict','Category']]
    data_set['Dates'] = data_set['Dates'].dt.hour
    return data_set

def get_column_x(data_set):
    return [i for i in data_set.columns if i not in ['Category']]

def get_column_y(data_set):
    return 'Category'

def fit_transform(data_set):
    Label_type = preprocessing.LabelEncoder()

    crime_type = Label_type.fit_transform(data_set.Category)
    Dates = Label_type.fit_transform(data_set.Dates)
    DayOfWeek = Label_type.fit_transform(data_set.DayOfWeek)
    dis = Label_type.fit_transform(data_set.PdDistrict)
    data_set['Dates'] = Dates
    data_set['DayOfWeek'] = DayOfWeek
    data_set['PdDistrict'] = dis
    data_set['Category'] = crime_type
    return data_set

def fit_bin_transform(y_train):
    return LabelBinarizer().fit_transform(y_train)


def fit_dict_transform(data_set,cls_name):
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

def get_data_set_x(data_set):
    return data_set[['Dates','DayOfWeek','PdDistrict']]

def get_data_set_y(data_set):
    return data_set['Category']