from numpy import *

def textParse(bigString) :
    import re
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest() :
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26) :
        wordList = textParse(open('email/spam/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocablist(docList)
    trainSet = list(range(50))
    testSet = []
    for i in range(10) :
        randIndex = int(random.uniform(len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet :
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    pSpam, p0v, p1v = trainNB0(array(trainMat), array(trainClass))
    exit
    errorCount = 0
    for docIndex in testSet :
        wordVec = setOfWords2Vec(vocabList, docList)
        if(classifyNB(array(wordVec), p0v, p1v, pSpam) != classList[docIndex]) :
            errorCount += 1

    print(errorCount)


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
            # returnVec[vocabList.index(word)] = 1
            returnVec[vocabList.index(word)] += 1
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

    #初始化为0有问题, 以为在[式2]中很容易出现分母0的情况
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

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

    #因为分母和分子相差悬殊, 由于结果的对比值比较重要, 具体指不重要
    #所以取对数
    p0Vect = log(p0Num / p0Denom)
    p1Vect = log(p1Num / p1Denom)

    #p(c1), [式2-0], [式-1]
    return pAbusive, p0Vect, p1Vect
            
#这个地方有问题, 就是免去了计算p(w)的工作, 因为p1和p0都要除以p(w), 所以直接去掉
#由于p1Vec和pClass1都取了对数, 所以相乘的式子都变了相加和sum
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1) :
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0 :
        return 1
    else :
        return 0

#testing
def testingNB() :
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocablist(listOPosts)
    
    trainMat = []
    for postInDoc in listOPosts :
        trainMat.append(setOfWords2Vec(myVocabList, postInDoc))
    pAb, p0V, p1V = trainNB0(trainMat, listClasses)

    #class0 test
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) 
    ret = classifyNB(thisDoc, p0V, p1V, pAb)
    print(ret)

    #class1 test
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) 
    ret = classifyNB(thisDoc, p0V, p1V, pAb)
    print(ret)

spamTest()
