# coding= utf8

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
import operator

#创建数据源
def creat_dataset():
    datasets = array([[8,4,2],[7,1,1],[1,4,4],[3,0,5]])#数据集
    labels = ['非常热','非常热','一般热','一般热']#类标签
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
    print(sortdDistindexs)
    #3.针对k个点，统计各个类别的数量
    classCount = {}
    for i in range(k):
        #根据距离排序索引值找到类标签
        votelabel = labels[sortdDistindexs[i]]
        print(sortdDistindexs[i],votelabel)
        #统计类标签的键值对
        classCount[votelabel] = classCount.get(votelabel,0)+1
        print(classCount)
    #4.投票机制，少数服从多数原则
    #对各个分类字典进行排序，降序，按照值排序
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(0))
    print(sortedclassCount)
    #print(newV,'KNN投票预测结果是：',sortedclassCount[0][0])
    return sortedclassCount[0][0]
#欧式距离计算1
#def Computeeuclideandistance(x1,x2,y1,y2):
    #d = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
    #return d

#欧式距离计算2
#def Euclideandistance(instance1,instance2,length):
    #d = 0
    #for i in range(length):
        #d += pow((instance1[i]-instance2[2]),2)
    #return math.sqrt(d)

#欧式距离计算3，这个比较常用，掌握
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

if __name__ == '__main__':
    #1.创建数据集和类标签
    datasets,labels = creat_dataset()
    print('数据集:\n',datasets,'\n类标签:\n',labels)

    #2.可视化分析数据
    #analyze_data_plot(datasets[:,0], datasets[:,1])

    #3.1欧式距离计算1
    #des1 = Computeeuclideandistance(2,4,8,2)
    #print('欧式距离计算1:',des1)

    #3.2欧式距离计算2
    #des2 = Euclideandistance([2,4,4],[7,1,1],3)
    #print('欧式距离计算2:',des2)

    # 3.3欧式距离计算3
    #des3 = Euclideandistance3([2,4,4], datasets)
    #print('欧式距离计算3:',des3)

    # 4.1 单实例构造KNN分类器
    newV = [2,4,4]
    res = knn_Classifier(newV, datasets, labels, 3)
    print(newV, 'KNN投票预测结果是：', res)

    #4.2 多实例构造KNN分类器
    # vecs = array([[2,4,4],[3,0,0],[5,7,2]])
    # for vec in vecs:
    #     res = knn_Classifier(vec,datasets,labels,3)
    #     print(vec, 'KNN投票预测结果是：', res)





