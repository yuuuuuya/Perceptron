import numpy as np
import pdb#pdb.set_trace()
import os.path
import importlib#https://qiita.com/mriho/items/52f53559ba7fe7ef06ff


'''
all elements of vector to be exponential value.
input  : vector
output : exponential vector.
'''
def vecExp(vec):
    row = vec.shape[0]
    for i in range(row):
        val = vec[i]
        expval = np.exp(val)
        vec[i] = expval
    return vec
'''
each elements are each probability funtion that correspond label.
input  : input vector(1,784), parameter W(10,784)
output : vector that elements are probability function that correspond label.
'''
def softmaxVec(inputVec, paraW, coeffi, penaModel):
    vec = np.dot(paraW,inputVec)#array shape is (10,1)
    penaVec = penaltyVec(paraW, coeffi, penaModel)
    plus_penaVec = vec + penaVec
    moveVec = move(plus_penaVec)
    expVec = vecExp(moveVec)
    denominator = np.sum(expVec)
    softmaxVec = expVec / denominator
    return softmaxVec

'''
softmax function.
input  : inputVec(1,784), parameterW(10,784), outputVal(scalor(0~9))
output : value of each label's plobability model(function)
'''
def softmax(inputVec, paraW, outputVal, coeffi, penaModel):
    #vector that elements are probability function
    softmax_Vec = softmaxVec(inputVec, paraW, coeffi, penaModel)
    softmaxVal = softmax_Vec[outputVal]
    return softmaxVal

'''
move the elements of vector for prevent to large np.exp()
input  : vector
output : vector that mean is 0
'''
def move(vec):
    minval = np.amin(vec)
    maxval = np.amax(vec)

    #New_vec = vec - minval#all elements > 0
    #norm = (maxval - minval)/2#mean to be 0
    #object_vec = New_vec - norm

    object_vec = vec -((minval + maxval)/2)

    return object_vec

'''
value of softmax function to be log value.
input  : inputVec(1,784), paraW(9,784), outputVal(scalor(0~9))
output : log value that softmax function value to be log value.
'''
def log_softmax(inputVec, paraW, outputVal, coeffi, penaModel):
    val = softmax(inputVec, paraW, outputVal, coeffi, penaModel)
    if val ==0:
        pdb.set_trace()
    logVal = np.log(val)
    return logVal

'''
likehood function. (to do riobbins monrroe)
plus log_softmax function only number of data.
input  : inputArray(60000,784), parameter W(10,784), output vector(60000,1)
output : likehood value.
'''
def likehoodFun(inputArray, paraW, outputVec, coeffi, penaModel):
    row, col = inputArray.shape#(60000,784)
    val = 0
    for i in range(row):
        inputVec = inputArray[i]
        outputVal = outputVec[i]
        softmax_log = log_softmax(inputVec, paraW, outputVal, coeffi, penaModel)
        val = val + softmax_log
    return val
'''
common part of diff value.
input  : input vector(1,784), paramter W(10,784)
output : array(common part of diff value)(10,784)
'''
def commonDiff(inputVec, paraW, coeffi, penaModel):
    dataDim, dataNum = paraW.shape
    penaltyVec = inputVec + diff_penaltyModel(inputVec, coeffi, penaModel)
    softVec = softmaxVec(inputVec, paraW, coeffi, penaModel)
    record = np.zeros((dataDim, dataNum))

    for i in range(dataDim):
        commonVec = -(penaltyVec*softVec[i])
        record[i] = commonVec
    return record

'''
not common part of diff value.
input  : input vector(1,784), parameter W(10,784), outputval(label = 0~9)
output : array(not common part of diff value)(10,784)
'''
def Not_commonDiff(inputVec, paraW, outputVal, coeffi, penaModel):
    row, col = paraW.shape
    record = np.zeros((row, col))
    record[outputVal] = inputVec + diff_penaltyModel(inputVec, coeffi, penaModel)
    return record

'''
Derivative of log_softmax function with respect to W
input  : input vector(1,784), parameter W(10,784), outputVal(label = 0~9)
output : diff value in the type of array.
'''
def diff_softmax(inputVec, paraW, outputVal, coeffi, penaModel):
    common = commonDiff(inputVec, paraW, coeffi, penaModel)
    commonNot = Not_commonDiff(inputVec, paraW, outputVal, coeffi, penaModel)
    arrayDiff = common + commonNot
    return arrayDiff

'''
deriverative of likehood function with respect to parameter W. (to do riobbins monrroe)
plus diff_softmax_medel function only number of data.
input  : inputArray(60000,784), output vector(60000,1), parameter W(10,784)
output : diff likehood value with respect to parameter W.
'''
def diff_likehoodFun(inputArray, outputVec, paraW, coeffi, penaModel):
    dataNum, dimNum = inputArray.shape
    paraRow, paraCol = paraW.shape
    Record_array = np.zeros((paraRow, paraCol))

    for i in range(dataNum):
        inputVec = inputArray[i]
        outputVal = outputVec[i]
        diff_array = diff_softmax(inputVec, paraW, outputVal, coeffi, penaModel)
        Record_array = Record_array + diff_array

    return Record_array

'''
only element that Designated row and col is h(small value). other elements are 0.
input  : array , element number that want to be h(small value).
output : array that only designated element is h(small value)
'''
def h_array(array, rowNum, colNum, h):
    row, col = array.shape
    recordarray = np.zeros((row, col))

    recordarray[rowNum,colNum] = h

    return recordarray

'''
confirming diff value. (Derivative with respect to W)
input  : inputdata's vector(784,1), parameter W(10,784), output value (scalor), h(small value)
output : Derivate of log_softmax with respect to parameter W
'''
def confirmDiff(inputVec, paraW, outputVal, h):
    rowNum, colNum = paraW.shape
    paraRecord = np.zeros((rowNum, colNum))

    for i in range(rowNum):
        row = i

        for j in range(colNum):

            plus_array = h_array(paraW, row, j, h)

            h_val = log_softmax(inputVec, paraW + plus_array, outputVal)
            val = log_softmax(inputVec, paraW, outputVal)

            diff_val = (h_val - val)/h
            paraRecord[row,j] = diff_val

    return paraRecord

'''
Normalization model (ridge regression and lasso regression)
input  : vector(parameter)
output : scalor(one of the penlty. )
'''
def ridge(inputVec):
    penalty = np.dot(inputVec,inputVec)#penalty is squre(jijou) norm of prameter W
    return penalty

def lasso(inputVec):
    factorial = np.dot(inputVec,inputVec)#factorial is kaijou
    penalty = np.sqrt(factorial)#penalty is norm of poarameter W
    return penalty

'''
penalty model(penalty is ridge regression or lasso regression)
input  : vector(parameter), Regularization coefficient(keisuu), normalization model
output : scalor(penalty for ordering to nomalize)
'''
def penaltyModel(inputVec, coeffi, penaModel):
    if penaModel=="ridge":
        val = ridge(inputVec)
    elif penaModel =="lasso":
        val = lasso(inputVec)
    else:
        raise NotImplementedError
    penalty = (val)*(coeffi)
    return penalty

'''
penalty in type of veector.
penalty of first element is first row of parameterW(10,784)'s penalty.
input  : parameter W(10,784), coefficient, penalty model('ridge or lasso')
output : penalty vector.
'''
def penaltyVec(paraW, coeffi, penaModel):
    paraRow, paraCol = paraW.shape
    recordVec = np.zeros((paraRow,))
    for i in range(paraRow):
        paraVec = paraW[i,:]
        penalty = penaltyModel(paraVec, coeffi, penaModel)
        recordVec[i] = penalty
    return recordVec

'''
derivative funstion of Normalize model(redge regression or lasso regression)
input  : vector(parameter)
output : vector(derivative value)
ridge_diff output is 2*W(parameter). lasso_diff output's element is if wi > 0 => 1, if wi < 0 => -1
'''
def ridge_diff(inputVec):
    penaltyGrad = 2*inputVec
    return penaltyGrad

def lasso_diff(inputVec):
    pdb.set_trace()
    penaltyGrad = np.sign(inputVec)
    return penaltyGrad
'''
select penalty diff modele. moreover, output is penalty((penalty model value)*(coefficient))
input  : input vector(1,784), coefficient, penalty modele.
output : penalty value(vector : (1,784))
'''

def diff_penaltyModel(inputVec, coeffi, penaModel):
    if penaModel=="ridge":
        val = ridge_diff(inputVec)
    elif penaModel =="lasso":
        val = lasso_diff(inputVec)
    else:
        raise NotImplementedError
    penalty = (val)*(coeffi)
    return penalty
