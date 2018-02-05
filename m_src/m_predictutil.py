import numpy as np
import m_funcutil as m_fun
import pdb

'''
predict inputdata in the type of vector. predict value is 0~9
input  : inputVec(one of the inputdata(1,784)), trainparaW(trained paraW (10,784))
output : predict value (scalor 0~9)
'''
def predict_vec(inputVec, trainparaW, coeffi, penaModel):
    classNum, dimNum = trainparaW.shape

    predict_vec = []
    epochIdx = 0

    for i in range(classNum):
        predictVal = m_fun.softmax(inputVec, trainparaW, i, coeffi, penaModel)
        predict_vec = predict_vec + [predictVal]
        epochIdx += 1

    Max = np.amax(predict_vec)
    #output is place of Max(0~9)
    predict = np.where(predict_vec == Max)[0]

    return predict

'''
predict inputdata in the type of array. predict value is 0~9, and output type is vector
input : inputdata, trainparaW(trained paraW (10,784))
output : predict vector. this vector elements are predict value of each inputdata(vector).
'''
def predict_array(inputArray, trainparaW, coeffi, penaModel):
    dataNum, dimNum = inputArray.shape
    record = np.zeros((dataNum,1))

    for i in range(dataNum):
        inputVec = inputArray[i]
        predictVal = predict_vec(inputVec, trainparaW, coeffi, penaModel)
        record[i] = predictVal

    return record

'''
reseach correct persentage.
input  : predicted vector, test's outputdata in the type of vector.
output : correct persentage. (scalor)
'''
def accuracy(predictVec, t_outputVec):
    dataNum = predictVec.shape[0]

    judge_vec = predictVec - t_outputVec
    correctlist = np.where(judge_vec == 0)[0]
    correctNum = len(correctlist)

    persentage = correctNum / dataNum

    return persentage
