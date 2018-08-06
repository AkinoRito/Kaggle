import csv
import numpy as np
import operator


def toInt(array):
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def normalizing(array):
    m, n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i][j] > 0:
                array[i][j] = 1
    return array


def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001 * 785
    l.remove(l[0])
    l = np.array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return normalizing(toInt(data)), toInt(label)


def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return normalizing(toInt(data))


# result 是结果列表，csvName 是存放结果的 csv 文件名
def saveResult(result, csvName):
    with open(csvName, 'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)


# 调用scikit的knn算法包
from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData, trainLabel, testData):
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData, trainLabel.values.ravel())
    testLabel = knnClf.predict(testData)
    saveResult(testLabel, 'sklearn_knn_result.csv')
    return testLabel


# 调用scikit的SVM包
from sklearn import svm
def svmClassify(trainData, trainLabel, testData):
    svmClf = svm.SVC()
    svmClf.fit(trainData, trainLabel.values.ravel())
    testLabel = svmClf.predict(testData)
    saveResult(testLabel, 'sklearn_svm_result.csv')
    return testLabel


# 调用scikit的朴素贝叶斯算法包，GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB
def GaussianNBClassify(trainData, trainLabel, testData):
    nbClf = GaussianNB()
    nbClf.fit(trainData, trainLabel.values.ravel())
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_GaussianNB_result.csv')
    return testLabel


from sklearn.naive_bayes import MultinomialNB
def MultinomialNBClassify(trainData, trainLabel, testData):
    nbClf = MultinomialNB()
    nbClf.fit(trainData, trainLabel.values.ravel())
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_MultinomialNB_result.csv')
    return testLabel


def digitRecognition():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    # 使用不同算法
    result1 = knnClassify(trainData, trainLabel, testData)
    result2 = svmClassify(trainData, trainLabel, testData)
    result3 = GaussianNBClassify(trainData, trainLabel, testData)
    result4 = MultinomialNBClassify(trainData, trainLabel, testData)


if __name__ == '__main__':
    digitRecognition()
