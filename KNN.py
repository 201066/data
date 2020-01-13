# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:14:17 2020

@author: 2010
"""

import numpy as np
import operator

"""
函数说明:创建数据集
group - 数据集
labels - 分类标签
"""

def createDataSet():
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    labels = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    return group,labels

"""
函数说明:k-近邻算法
inX - 用于分类的数据(测试集)
dataSet - 用于训练的数据(训练集)
labes - 分类标签
k - kNN算法参数,选择距离最小的k个点
返回:
    sortedClassCount[0][0] - 分类结果
"""

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]  #shape[0]表示dataSet的行数
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet  #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    sqDiffMat = diffMat**2  #差的平方
    sqDistances = sqDiffMat.sum(axis=1) #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    distances = sqDistances**0.5 #开方，计算出距离
    sortedDistIndices = distances.argsort() #返回distances中元素从小到大排序后的索引值
    classCount = {}
    for i in range(k): #取出前k个元素的类别    
        voteIlabel = labels[sortedDistIndices[i]]
        
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
                                                                    #计算类别次数
    
    
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #reverse降序排序字典
    
    return sortedClassCount[0][0]  #返回次数最多的类别,即所要分类的类别
if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [18,90]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)
