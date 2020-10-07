from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    # inX测试向量，dataSet训练样本集，labels训练样本对应的标签，k最近邻数目
    dataSetSize = dataSet.shape[0]  # shape[0]求行数
    # 求欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile(A,(m,n))用A构造m行n列
    sqdiffMat = diffMat**2
    sqDistances = sqdiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = argsort(distances)  # 返回排序后索引
    classCount = {}  # 空字典
    # 选择k个最近邻
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #统计k近邻类的数目

    sortClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortClassCount[0][0]


def img2vector(filename):
    rows = 32
    cols = 32
    imgVector = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(rows):
            lineStr = fr.readline()
            for j in range(cols):
                imgVector[0, i*32+j] = int(lineStr[j])
    return imgVector


def vector2mat():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))  # 训练集合并成大矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)  # 类别
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    return trainingMat, hwLabels


def main():
    # 分类
    trainingMat, hwLabels = vector2mat()

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        testFileName = testFileList[i]
        testFileVector = img2vector('testDigits/%s' % testFileName)
        testClass = int(testFileName.split('.')[0].split('_')[0])
        k = 3  # 设置最近邻数目
        predictClass = classify0(testFileVector, trainingMat, hwLabels, k)
        print("classifier come back with %d, the real answer is %d\n" % (testClass, predictClass))
        if(predictClass != testClass):
            errorCount += 1.0
    print("the number of errors is %d\n" % errorCount)
    print("the rate of error is %f\n" % (errorCount/float(mTest)))


if __name__ == '__main__':
    main()