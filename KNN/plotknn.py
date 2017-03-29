# -*- coding: utf-8 -*-

import KNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
'''
###  reload ###
reload()函数将以前导入过的模块再加载一次。重新加载（reload）包括最初导入模块时应用的分析过程和初始化过程。
这样就允许在不退出解释器的情况下重新加载已更改的Python模块。
这里事实上不需要reload。
'''
reload(KNN)

'''
本例子所用的数据集包括三种特征，并分为三类。这里将第二和第三种特征作为坐标，在散点图中表示出分类情况依据该坐标的分布情况
'''
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet2.txt')
datingDataMat, ranges, minVals = KNN.autoNorm(datingDataMat)  # 当函数返回多个值时，若只给一个变量，那么将得到tuple类型
datingLabels = map(int, datingLabels)  # map函数 对列表中的每个元素都使用同样的元素操作，比如说这里的int()函数，转换字符串为整形
fig = plt.figure()
ax = fig.add_subplot(111)   # 将画布割成1行1列取第一块
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))  # scatter函数 画散点图，给定横纵坐标的列表，第一个数组表示点的大小，第二个坐标表示颜色（具体还不清楚）
plt.show()