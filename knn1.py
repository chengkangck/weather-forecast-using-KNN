# coding= utf8

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
import operator

#创建数据源1
def creat_dataset():
    datasets = array([[8,4,2],[7,1,1],[1,4,4],[3,0,5]])#数据集
    labels = ['非常热','非常热','一般热','一般热']#类标签
    return datasets,labels

#创建数据源2
def creat_dataset2():
    datasets = array([[8,4,2],[7,1,1],[1,4,4],[3,0,5],[9,4,2],[7,0,1],[1,5,4],[4,0,5]])#数据集
    labels = ['非常热','非常热','一般热','一般热','非常热','非常热','一般热','一般热']#类标签
    return datasets,labels

def analyze_data_plot(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    plt.show()

#构造KNN分类器
def knn_Classifier(newV,datasets,labels,k):
    #1.计算样本数据与样本库函数之间的距离
    sqrtDist = Euclideandistance3(newV,datasets)
    #2.根据距离进行排序
    sortdDistindexs = sqrtDist.argsort(axis=0)
    # print(sortdDistindexs)
    #3.针对k个点，统计各个类别的数量
    classCount = {}
    for i in range(k):
        #根据距离排序索引值找到类标签
        votelabel = labels[sortdDistindexs[i]]
        # print(sortdDistindexs[i],votelabel)
        #统计类标签的键值对
        classCount[votelabel] = classCount.get(votelabel,0)+1
        # print(classCount)
    #4.投票机制，少数服从多数原则
    #对各个分类字典进行排序，降序，按照值排序
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(0))
    # print(sortedclassCount)
    #print(newV,'KNN投票预测结果是：',sortedclassCount[0][0])
    return sortedclassCount[0][0]

#欧式距离计算3
def Euclideandistance3(newV,datasets):
    #获取数据向量维度值
    rowsize,closize = datasets.shape
    #各特征向量求差值
    diffMat = tile(newV, [rowsize, 1]) - datasets
    # print(tile(newV,[rowsize,1]))
    # print(diffMat)
    #对插值平方
    sqdiffMat = diffMat**2
    # print(sqdiffMat)
    #差值平方和进行开方
    sqrtDist = sqdiffMat.sum(axis=1)**0.5
    return sqrtDist

def predict_temperature():
    datasets, labels = creat_dataset()
    iceCream = eval(input('请问你今天吃了几个冰淇淋？\n'))
    drinkWater = eval(input('请问你今天喝了几杯水？\n'))
    playTime = eval(input('请问你今天户外活动几小时？\n'))
    newV = [iceCream,drinkWater,drinkWater]
    res = knn_Classifier(newV, datasets, labels, 3)
    print('该访客认为成都天气是：', res)

if __name__ == '__main__':

    # 4.1 单实例构造KNN分类器
    # newV = [2,4,4]
    # res = knn_Classifier(newV, datasets, labels, 3)
    # print(newV, 'KNN投票预测结果是：', res)

    #4.2 多实例构造KNN分类器
    # vecs = array([[2,4,4],[3,0,0],[5,7,2]])
    # for vec in vecs:
    #     res = knn_Classifier(vec,datasets,labels,3)
    #     print(vec, 'KNN投票预测结果是：', res)

    # 5 利用KNN分类器预测随机访客天气感知度
    predict_temperature()
