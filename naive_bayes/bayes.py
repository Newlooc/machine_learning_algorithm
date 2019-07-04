

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

def createVocablist(dataSet) :
    vocabSet = set([])
    for document in dataSet :
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet) :
    returnVec = [0] * len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
    return returnVec

data, label = loadDataSet()
vocab = createVocablist(data)
res = setOfWords2Vec(vocab, data[2])
print(res)
