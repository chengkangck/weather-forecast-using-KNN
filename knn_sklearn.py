# coding= utf8

from sklearn import neighbors
import knn1
from numpy import *

def knn_sklearn_predict(newV, datasets, labels):
    #调用机器学习库knn分类器算法
    knn = neighbors.KNeighborsClassifier()

    #传入参数，特征数据，分类标签
    knn.fit(datasets,labels)
    #knn预测
    predictsRes = knn.predict([newV])

    print('该访客认为成都天气是：', predictsRes)
    return predictsRes

def predict_temperature():
    #调用knn1模块下的创建数据集方法，返回数据特征集和类标签
    datasets,labels = knn1.creat_dataset2()
    iceCream = eval(input('请问你今天吃了几个冰淇淋？\n'))
    drinkWater = eval(input('请问你今天喝了几杯水？\n'))
    playTime = eval(input('请问你今天户外活动几小时？\n'))
    newV = [iceCream,drinkWater,drinkWater]
    knn_sklearn_predict(newV, datasets, labels)

if __name__ == '__main__':
    predict_temperature()