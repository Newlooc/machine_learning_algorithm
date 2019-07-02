from math import log
import operator

def calShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet :
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys() :
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts :
        #计算几率
        prob = float(labelCounts[key]) / numEntries
        #计算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet() :
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
    
#取以特征列向量为维度划分(根据value)的数据集合
def splitDataSet(dataSet, axis, value) :
    retDataSet = []
    for featVec in dataSet :
        if featVec[axis] == value:
            #取特征列向量前的元素
            reducedFeatVec = featVec[:axis]
            #取特征列向量后的元素并拼接
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 获得划分dataSet的最优(信息熵最小或信息熵递减增益最大)Feature
def chooseBestFeatureToSplit(dataSet) :
    #获取单笔数据的长度(去label)
    numFeature = len(dataSet[0]) - 1
    #原始无序数据的信息熵
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    #遍历所有的feature
    for i in range(numFeature) :
        #取feature i在所有data中的值, 形成列向量
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算新的有序集的信息递减增益
        for value in uniqueVals :
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob *  calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #返回增益最大的feature
        if(infoGain > bestInfoGain) :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
    
#获取数量最多的分类
def majorityCnt(classList) :
    classCount = {}
    for vote in classList :
        if vote not in classCount.keys() :
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels) :
    classList = [example[-1] for example in dataSet]
    if(classList.count(classList[0]) == len(classList)) :
        return classList[0]
    if(len(dataSet[0]) == 1) :
        return majorityCnt(dataSet)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    del(labels[bestFeat])
    featValue = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValue)
    for value in uniqueValues :
        subLabel = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabel)
    return myTree
        

dataSet, labels = createDataSet()
calShannonEnt(dataSet)
chooseBestFeatureToSplit(dataSet)
res = createTree(dataSet, labels)
print(res)
