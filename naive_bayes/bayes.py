from numpy import *

#创建数据
def loadDataSet() :
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problem', 'help', 'pleases'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

#创建词汇列表, 将所有出现的单词整理起来
def createVocablist(dataSet) :
    #集合操作,为了去重
    vocabSet = set([])
    for document in dataSet :
        #全部使用Set(每个值都是唯一的)然后求并
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#按照给定的字典, 将文本转换成向量
def setOfWords2Vec(vocabList, inputSet) :
    returnVec = [0] * len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
    return returnVec

#求贝叶斯公式里的各个分项
#由于假设现在的贝叶斯公式是Naive的.所以, 
#一个句子向量的各个元素是独立的, 所以概率
#p(w|c_1)可以写成p(w_1|c_1)p(w_2|c_1)...p(w_n|c_1)[式1]
def trainNB0(trainMatrix, trainLabel) :
    #Data有多少笔
    numTrainDocs = len(trainMatrix)
    #单个Post转换成向量的维数(元素个数)
    numWords = len(trainMatrix[0])
    #负面评论几率
    pAbusive = sum(trainLabel) / float(numTrainDocs)
    #[式1]使用矩阵运算会更加方便, 所以使用numpy库
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs) :
        #二元分类, 直接用if...else
        if trainLabel[i] == 1 :
            #所有负面评论的Post的向量之和, 即各个分量出现的次数之和的向量
            p1Num += trainMatrix[i]
            #所有负面评论总词数
            p1Denom += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #向量[p(w_1|c_i), p(w_2|c_i), ... ,p(w_n|c_i)] [式2-i]
    p0Vect = p0Num / p0Denom
    p1Vect = p1Num / p1Denom
    #p(c1), [式2-0], [式-1]
    return pAbusive, p0Vect, p1Vect
            


data, label = loadDataSet()
vocab = createVocablist(data)
trainMat = []
for postinDoc in data :
    trainMat.append(setOfWords2Vec(vocab, postinDoc))

res = trainNB0(trainMat, label)
print(res)
