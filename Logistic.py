# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:12:04 2019

@author: 2010
"""

import numpy as np
import matplotlib.pyplot as plt



"""
函数说明：加载数据集
dataMat : 数据
labelMat : 标签
"""


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        line = line.strip().split()
        labelMat.append(int(line[-1]))
        dataMat.append([1.0,float(line[0]),float(line[1])])   #testSet的前连个值是X1,X2,为了方便计算，所以把x0设为1.0
    fr.close()
    return dataMat,labelMat

"""
函数说明：绘制数据集
"""
def plotDataSet():
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('x'); plt.ylabel('y')                                    #绘制label
    plt.show()      

"""
函数说明：sigmoid函数
x : 数据
"""

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

"""
函数说明：梯度下降算法，更新w
"""

def gradDescent(dataMat,classLabels):
    dataMatrix = np.mat(dataMat)   #shape:[100,3]
    labelMat = np.mat(classLabels).transpose()  #shape:[1,100]  label.t:[100,1]
    m,n = np.shape(dataMatrix)
    alpha = 0.001  #学习率
    maxCycles = 500  #迭代次数
    weights = np.ones((n,1))  #初始化 w
    for k in range(maxCycles):
        y_ = sigmoid(dataMatrix * weights)
        error = y_ - labelMat
        weights = weights - alpha*dataMatrix.transpose()*error
    return weights.getA()   #  返回求得的权重数组(最优参数)  

"""
函数说明：绘制决策边界
"""    
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()       

if __name__=="__main__":
    dataMat,labelMat = loadDataSet()
    plotDataSet()
    weights = gradDescent(dataMat,labelMat)
    plotBestFit(weights)
   
    
    
    
    
    
    