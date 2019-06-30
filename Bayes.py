# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Bayes
   Description :
   Author :        chen
   date：          2019/6/28
-------------------------------------------------
   Change Activity:
                   2019/6/28:
-------------------------------------------------
"""
import pandas as pd


class Bayes:
    """
    贝叶斯类
    """
    def __init__(self,data_set,column_x,column_y):
        """
        构造函数
        :param dataset: dataFrame格式，需要受训练的数据集
        :param column_x: 指明条件属性列表(list)
        :param column_y: 指明分类属性(string)
        """
        self.data_set = data_set
        self.column_x = column_x
        self.column_y = column_y

    def calc_p_y(self,column_y_val):
        """
        计算标签某一类的概率 P(Y=val)

        :example
        p = bayes.calc_p_y('WARRANTS')

        :param column_y_val:标签值
        :return:
        """
        count = 0
        for index,row in self.data_set.iterrows():
            if row[self.column_y] == column_y_val:
                count += 1
        return count / self.data_set.size

    def calc_p_x_y(self,column_x_val,column_y_val):
        """
        计算在某个标签下对应条件属性值的概率

        :example
        p = bayes.calc_p_x_y([23,'Wednesday','NORTHERN'],'WARRANTS')

        :param column_x_val:条件属性值列表[ attr1_val,attr2_val,attr3_val ]
        :param column_y_val:标签属性 class_val
        :return:该类别的概率
        """
        count_y = 0.0
        count_x = {}
        for col_name in self.column_x:
            count_x[col_name] = 0.0

        for index,row in self.data_set.iterrows():
            if row[self.column_y] == column_y_val:
                count_y += 1
                for col_name,col_val in zip(self.column_x,column_x_val):
                    if row[col_name] == col_val:
                        count_x[col_name] += 1

        result = 1.0
        for col_name in self.column_x:
            result_x = count_x[col_name] / count_y
            result *= result_x
        return result

    def set_test_x(self,Dates,DayOfWeek,PdDistrict):
        """
        设置预测条件
        :param Dates: 小时
        :param DayOfWeek:星期
        :param PdDistrict: 地区
        :return:列表
        """
        return [Dates,DayOfWeek,PdDistrict]

    def predict_one(self,column_x_val):
        """
        计算每一类的最终概率,以及预测最终的概率

        example:
         dict,result = bayes.predict(column_x_value)

        :param column_x_val:条件属性值列表[ attr1_val,attr2_val,attr3_val ]
        :return:每个类别的概率的字典，概率最大的类别
        """
        # 定义一个保存所有类别的概率的字典
        Category_p_dict = {}
        for cate in self.data_set[self.column_y]:
            Category_p_dict[cate]=-1

        for key in Category_p_dict:
            p_y = self.calc_p_y(key)
            p_x_y = self.calc_p_x_y(column_x_val,key)
            Category_p_dict[key] = p_y * p_x_y

        # 得出可能性最大的类别
        result = max(Category_p_dict, key=Category_p_dict.get)

        return Category_p_dict,result

    def predict_all(self,test_set):
        """
        预测所有测试数据集
        :param test_set: 测试数据集
        :return: 正确率
        """
        count = 0.0
        plus = test_set.size
        for index, row in test_set.iterrows():
            column_x_value = self.set_test_x(Dates=row['Dates'], DayOfWeek=row['DayOfWeek'],
                                              PdDistrict=row['PdDistrict'])
            dict, result = self.predict_one(column_x_value)
            print('[{0}]预测完成...'.format(index))
            if (result == row['Category']):
                print('   预测正确!'.format(index))
                count += 1

        return count/plus
