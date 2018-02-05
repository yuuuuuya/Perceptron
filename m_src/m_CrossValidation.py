import numpy as np
import pdb#pdb.set_trace()
import os.path
import sys
sys.path.append('/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data')

import time
import argparse
import matplotlib.pyplot as plt

import LoadMNIST as load
import m_cross as cross
import m_funcutil as fun
import m_predictutil as pre
import m_gradientutil as grad

datapath = '../result_folder/'




m_datapath = '/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data'
imgTrain = load.readMNIST('train_img', m_datapath)
labTrain = load.readMNIST('train_label', m_datapath)




def main(args):

    start = time.time()

    splitsNum = args.splits
    normMode = args.normMode
    itera = args.iteration
    stepSize = args.stepsize
    myseed = args.seed
    penaMode = args.penaMode
    Lgridnum = args.Lgridnumber


    #load original data frame.
    m_datapath = '/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data'
    imgTrain = load.readMNIST('train_img', m_datapath)
    labTrain = load.readMNIST('train_label', m_datapath)

    print('[ Spliting into %s data ]'%(splitsNum))

    #first row of trainRecord and testRecord are first data(array) of cross validation.
    trainRecord, testRecord = cross.CroVali_Data(imgTrain, labTrain, splits = splitsNum, normMode = normMode)

    coe_parameterGrid = np.linspace(0.0, 1.0, Lgridnum + 1)
    accAve_Record = []
    epochIdx_plotAcc = 1

    bestparaRecord = []
    epochIdx_bestpara = 1

    print('Start Cross Validation')

    for j in range(Lgridnum + 1):
        coeffi = coe_parameterGrid[j]

        accuracyRecord = []
        epochIdx_accu = 1

        paraRecord = []
        epochIdx_para = 1

        for i in range(splitsNum):
            inputTrain, outputTrain = trainRecord[i]
            inputTest, outputTest = testRecord[i]

            #make parameterW (10,7843)
            np.random.seed(myseed)
            paraW = np.random.normal(0, 0.01, size = (10,784))

            print('[ %s th training : number of data is %s ]'%(i, outputTrain.shape[0]))
            print('[ coefficient of regularization is %s ]'%(coeffi))

            #trainig (finding best parameters to be hight Likelihood(yuudo) function.)
            Erecord, train_paraW = grad.gradient(inputTrain, outputTrain, paraW, stepSize, itera, coeffi, penaMode)
            #plot energy for comfirming whether gradient method is correct.
            plt.plot(Erecord)

            #predict number (0~9), predict's shape is (number of data,1)
            predictVec = pre.predict_array(inputTest, train_paraW, coeffi, penaMode)
            #output is corect persent.
            accuracy = pre.accuracy(predictVec, outputTest)

            print('%s th cross validation accuracy : %s'%(i,accuracy))


            #memory several pasentage of correct answers
            accuracyRecord = accuracyRecord + [accuracy]
            epochIdx_accu += 1
            paraRecord = paraRecord + [train_paraW]
            epochIdx_para += 1


        #average of pasentage of correct answers
        average = np.sum(accuracyRecord)/splitsNum
        print('accuracy average:%s'%average)

        #find best parameterW in closs validation fixed coefficient of regularization.
        accuMax = np.amax(accuracyRecord)
        accuMax_place = np.where(accuracyRecord == accuMax)[0][0]
        bestpara = paraRecord[accuMax_place]
        #record best parameterW in closs validation fixed coefficient of regularization.
        bestparaRecord = bestparaRecord + [bestpara]
        epochIdx_bestpara += 1

        #record accuracy average on each coefficient of regularization.
        accAve_Record = accAve_Record + [average]
        epochIdx_plotAcc += 1

    end = time.time()
    print("Penalty mode: %s, Data normalization mode:%s"%(penaMode, args.normMode))
    print(u"演算所要時間%s秒"%(end-start))

    #find best accuracy average value.
    bestAccuAverage = np.amax(accAve_Record)
    bestpara_place = np.where(accAve_Record == bestAccuAverage)[0][0]
    #find best parameter in the all
    bestpara_All = bestparaRecord[bestpara_place]
    bestCoffi = coe_parameterGrid[bestpara_place]


    figname = args.figname + '_step%s_iter%s_%s_lamb%s'%(stepSize, itera, penaMode, bestCoffi)

    #energy plot figure.
    figpathHistory = os.path.join(datapath, figname + 'Ene_history.png')
    plt.legend()
    plt.savefig(figpathHistory)
    plt.close()

    #accuracy average plot figure on each coefficient of regularization.
    figpathAccuHistory = os.path.join(datapath, figname + 'Accu_history.png')
    plt.plot(accAve_Record)
    plt.xlim(0.0,1.0)
    plt.title("Cross Validation Normalize Model:%s"%(penaMode))
    plt.xlabel("Normalization Coefficient")
    plt.ylabel("Accuracy average")
    plt.savefig(figpathAccuHistory)

    #save best parameter
    parname = args.parname + '_cross_step%s_iter%s_%s_lamb%s.npy'%(stepSize, itera, penaMode, bestCoffi)
    parameterNamePath = os.path.join(datapath, parname)
    np.save(parameterNamePath, bestpara_All)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Cross Valisation')


    parser.add_argument('--figname', '-fig', type =str, default = 'cross' )
    parser.add_argument('--parname', '-par', type =str, default = 'para' )
    parser.add_argument('--iteration', '-i', type =int, default =200 )
    parser.add_argument('--stepsize', '-s', type =float, default = 0.0000007 )
    #normMode:'Max','Norm','Laplace'
    parser.add_argument('--normMode', '-n', type = str, default = 'Laplace' )
    parser.add_argument('--splits', '-spl', type = int, default = 5 )
    parser.add_argument('--seed', '-seed', type =int, default = 3, help = 'fixed random')
    #'ridge','lasso'
    parser.add_argument('--penaMode', '-pena', type = str, default = 'ridge', help = 'penalty model(regularization)')
    parser.add_argument('--Lgridnumber', '-lg', type = int, default = 10, help = 'lamb grid number')


    args = parser.parse_args()

    main(args)
    os.system('say "分割交差検証終了"')
    os.system('open -a Finder %s'%datapath)
