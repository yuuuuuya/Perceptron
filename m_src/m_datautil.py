import numpy as np
import pdb
import sys
sys.path.append('/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data')

import LoadMNIST as load

datapath = '/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data'
imgTrain = load.readMNIST('train_img', datapath)
labTrain = load.readMNIST('train_label', datapath)


'''
reseach number of each label data.
The form of the output is list.
[0 data Num, ... ,9 data Num]
input  : labTrain(outputdata), labNum(number of class)
output : list(number of each label data)
'''
def dataNumlist(labTrain, classNum):
    list = []
    epochIdx = 0
    for i in range(classNum):
        val = labTrain[labTrain==i].shape[0]
        list = list + [val]
    return list

'''
shuffle the rows of the array.
input  : trainData(trainData that combines imgTrain and labTrain in the horizontal(yoko) direction)
output : shuffled array in the horizontal(yoko) direction.
'''
def shuffle(trainData):
    shuffled_array = np.take(trainData,np.random.permutation(trainData.shape[0]),axis=0)
    return shuffled_array

'''
divide tranDara into inputdata and outputdata.
input  : traindata(trainData that combines imgTrain and labTrain in the horizontal(yoko) direction)
output : inputdata(array(data num,784)), outputdata(vector(data num, 1))
'''
def divide(trainData):
    row, col = trainData.shape
    inputData =  trainData[:,0:col-1]
    outputData = trainData[:,col-1]
    return inputData, outputData

'''
matcth the number of eath label(0~9) datas.
input  : imgTrain(inputdata), labTrain(outputdata), classNum(the number of class)
output : datamatch(matched the number of each label datas. shuffled row in the horizontal(yoko) direction.
this array that combine inputdata and outpuytdata)
'''
def dataMatch(imgTrain, labTrain, classNum):
    Numlist = dataNumlist(labTrain, classNum)
    minNum = np.amin(Numlist)

    trainData = np.c_[imgTrain,labTrain]
    traRow, traCol = trainData.shape

    epochIdx = 0
    trainDataRecord = []
    for i in range(classNum):
        # extract label = i datas
        extractData = trainData[trainData[:,traCol-1]==i]
        #shufle
        shuffleData = shuffle(extractData)
        #cut the data 0 ~ minNum(the number of the most small label data)
        cutdata = shuffleData[0:minNum,:]
        trainDataRecord = trainDataRecord + [cutdata]
        epochIdx =+1

    #combine trainDataRecord in the lengthwise(tate) direction
    combineData = np.concatenate(trainDataRecord, axis = 0)
    #shuffl
    datamatch =  shuffle(combineData)
    #divide datamatch array into inputdata and outputdata.
    inputData, outputData = divide(datamatch)

    return inputData, outputData

'''
Normalize input data
Input  : array(vector)
Output : array(normalized vector)
'''
#make Normalized array of max is 1 and min is 0 by odividing norm
def regularizeNorm(rawdata):
    v = rawdata/np.linalg.norm(rawdata)
    return v


def regularizeMax(rawdata):
    rawdataMax = np.max(rawdata)
    rawdataMin = np.min(rawdata)
    rawdataAve = np.mean(rawdata)
    rawdataDelta = rawdataMax - rawdataMin
    v = (rawdata - rawdataAve)/rawdataDelta
    return v

#make Normalized array of mean is 0 and variance is 1
def regularizeLaplace(rawdata):
    rawdataAve = np.mean(rawdata)
    rawdataStd = np.sqrt(np.var(rawdata))
    v = (rawdata - rawdataAve)/rawdataStd
    return v


'''
input  : array(number of data, number of types of data)
output : normalized array(number of data, number of types of data)
'''
def Normalization(inputdata, mode = 'Laplace'):
    row, col = inputdata.shape
    mylist = []
    for k in range(row):
        #Normalize k row by regularizeMax()
        if mode == 'Max':
            mylist += [regularizeMax(inputdata[k,:])]
        #Normalize k row by regularizeNorm()
        elif mode == 'Norm':
            mylist += [regularizeNorm(inputdata[k,:])]
        #Normalize k row by regularizeLaplace()
        elif mode == 'Laplace':
            mylist += [regularizeLaplace(inputdata[k,:])]
        else:
            raise NotImplementedError
    #np.c_() get rows combine
    norm_array = np.c_[mylist]
    return norm_array

'''
Cut out only the number of data required
input  : imgTrain(origin inputdata), labTrain(origin outputdata), dataNum(the number of data required)
output : inputdata(data required Num, 748), outputdata(data required Num,1)
'''
def maketrain(imgTrain, labTrain, dataNum):
    trainData = np.c_[imgTrain,labTrain]
    shuffledata = shuffle(trainData)
    cutdata = shuffledata[0:dataNum,:]

    inputData, outputData = divide(cutdata)

    return inputData, outputData
