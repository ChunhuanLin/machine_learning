# -*- coding: utf-8 -*-   #在文件的开头加上这行就可以写中文啦！

from numpy import *
import operator

'''
作为测试用的数据
'''
def createDataSet():

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile
    sqDiffMat = diffMat**2  # nparray.sum
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()  # argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1  # dict.get(),0 is the default value when key is none
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # sorted 第一个参数得到要排序的对象，第二个参数表示用排序对象的哪个域进行排序
    return sortedClassCount[0][0]

'''
三个特征，一个标签
从文本中读取

'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   # 去掉换行符
        listFromLine = line.split('\t')  # 去掉分隔符，返回列表
        returnMat[index, :] = listFromLine[0:3]   # 将一个列表的值赋值给了一个数组
        classLabelVector.append(listFromLine[-1])   # 列表扩展
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)    # 对一个数组按列取最小值，0表示按列取最小
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(maxVals-minVals, (m,1))

    return normDataSet, ranges, minVals

