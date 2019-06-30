# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :        chen
   date：          2019/6/29
-------------------------------------------------
   Change Activity:
                   2019/6/29:
-------------------------------------------------
"""
from Bayes import Bayes
from DTrees import DTrees
from neural_network import NeuralNetwork
from data_process import read_data,filter_data,fit_transform,get_column_x,get_column_y,fit_bin_transform,fit_dict_transform,get_data_set_x,get_data_set_y
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus
import graphviz

FILE_PATH = './dataset/train_sample.csv'
TEST_PATH = './dataset/test_sample.csv'

def bayes_predict():
    """贝叶斯分类预测"""
    data_set = read_data(FILE_PATH)
    data_set = filter_data(data_set)

    test_set = read_data(TEST_PATH)
    test_set = filter_data(test_set)

    column_x = get_column_x(data_set)
    column_y = get_column_y(data_set)

    bayes = Bayes(data_set,column_x,column_y)
    # column_x_value = bayes.set_test_x(Dates=23,DayOfWeek='Wednesday',PdDistrict='NORTHERN')
    # dict,result = bayes.predict(column_x_value)

    print('准备开始...')
    p = bayes.predict_all(test_set)
    print(p)

def dTrees_predict():
    """决策树分类预测"""
    data_set = read_data(FILE_PATH)
    data_set = filter_data(data_set)
    data_set = fit_transform(data_set)

    test_set = read_data(TEST_PATH)
    test_set = filter_data(test_set)
    test_set = fit_transform(test_set)

    column_x = get_column_x(data_set)
    column_y = get_column_y(data_set)

    dtrees = DTrees(data_set,test_set,column_x,column_y)

    train_x, train_y = dtrees.get_train_x_y()
    test_x,test_y = dtrees.get_test_x_y()

    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    # dot_data = tree.export_graphviz(model, out_file=None,
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    #
    # graph.render('example.gv', directory='.\\', view=True)

    predicted = model.predict(test_x)
    print("决策树准确度:", accuracy_score(test_y, predicted))

def neual_network_predict():
    """神经网络分类预测"""
    data_set = read_data(FILE_PATH)
    data_set = filter_data(data_set)
    data_set = fit_transform(data_set)

    test_set = read_data(TEST_PATH)
    test_set = filter_data(test_set)
    test_set = fit_transform(test_set)

    train_x = get_data_set_x(data_set)
    train_y = get_data_set_y(data_set)
    test_x = get_data_set_x(test_set)
    test_y = get_data_set_y(test_set)

    labels_train = fit_bin_transform(train_y)
    print(labels_train)
    network = NeuralNetwork([3,50,len(labels_train[0])])
    network.fit(train_x,labels_train,epochs=3000)
    a,b = network.predict_all(test_x, test_y)
    print(a,'\n',b)


def main():
    dTrees_predict()
    neual_network_predict()
    bayes_predict()

if __name__ == '__main__':
    main()