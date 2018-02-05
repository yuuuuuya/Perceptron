import numpy as np
import pdb#pdb.set_trace()
import os.path
import importlib#https://qiita.com/mriho/items/52f53559ba7fe7ef06ff
import m_funcutil as m_fun


def gradient(inputArray, outputVec, paraW, stepSize, loopNum, coeffi, penaModel):
    paraRow, paraCol = paraW.shape

    epochIdx_paraW = 0
    paraRecord = []
    Erecord = np.zeros(loopNum+1)

    Eval = m_fun.likehoodFun(inputArray, paraW, outputVec, coeffi, penaModel)
    paraWval = paraW

    paraRecord = paraRecord + [paraWval]
    epochIdx_paraW += 1
    Erecord[0] = Eval

    for i in range(loopNum):

        print('%s iterations complete \t: Current Energy is %s'%(i,Erecord[i]))
        gradW = m_fun.diff_likehoodFun(inputArray, outputVec, paraWval, coeffi, penaModel)
        updateW = gradW* stepSize
        paraWval = paraWval + updateW


        paraRecord = paraRecord + [paraWval]
        epochIdx_paraW += 1

        Erecord[i+1] = m_fun.likehoodFun(inputArray, paraWval, outputVec, coeffi, penaModel)


    train_paraW = paraRecord[loopNum-1]

    return Erecord, train_paraW
