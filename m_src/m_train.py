import numpy as np
import pdb#pdb.set_trace()
import os.path
import sys
sys.path.append('/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data')

import argparse
import time
import matplotlib.pyplot as plt

import LoadMNIST as load
import m_funcutil as fun
import m_gradientutil as grad
import m_datautil as data


datapath = '../result_folder/'


def main(args):

    start = time.time()

    dataNum = args.dataNum
    normMode = args.normMode
    itera = args.iteration
    stepSize = args.stepsize
    myseed = args.seed
    coeffi = args.coefficient
    penaMode = args.penaMode
    parname = args.parname

    print("Initiating the training sequence...")
    #make data(inputdata, outputdata, parameterW).
    #use data are 'norm_input', 'much_output', 'paraW'
    m_datapath = '/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data'
    imgTrain = load.readMNIST('train_img', m_datapath)
    labTrain = load.readMNIST('train_label', m_datapath)
    inputData, outputData = data.maketrain(imgTrain, labTrain, dataNum)
    match_input, match_output = data.dataMatch(inputData, outputData, classNum = 10)
    norm_input = data.Normalization(match_input, mode = normMode)
    print('[data size is %s]'%(match_output.shape))

    np.random.seed(myseed)
    paraW = np.random.normal(0, 0.01, size = (10,784))
    Erecord, train_paraW = grad.gradient(norm_input, match_output, paraW, stepSize, itera, coeffi, penaMode)



    parname = args.parname + '_train_step%s_iter%s_pena%s_lamb%s.npy'%(stepSize, itera, penaMode, coeffi)
    figname = args.figname + '_step%s_iter%s_%s_lamb%s'%(stepSize, itera, penaMode, coeffi)

    figpath = os.path.join(datapath, figname + '.png' )
    parameterNamePath = os.path.join(datapath, parname + '.npy')

    end = time.time()
    print(u"演算所要時間%s秒"%(end-start))
    print(u"parameter saved at %s"%parname)



    #plot upload values of likelihood function and save graph
    fig = plt.figure(figsize = (20,20) )
    plt.plot(Erecord)
    plt.title("Energy Transition (Number of data : %s, penalty model : %s, cefficient : %s)"%(match_output.shape,penaMode, coeffi))
    plt.xlabel("iterations")
    plt.savefig(figpath)

    #save finalparameters(W and a) on paramternamepath
    np.save(parameterNamePath, train_paraW)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Logistic regression')


    parser.add_argument('--parname', '-par', type =str, default = 'para' )
    parser.add_argument('--figname', '-fig', type =str, default = 'trajectory' )
    parser.add_argument('--dataNum', '-num', type =int, default = 58000 )
    parser.add_argument('--iteration', '-i', type =int, default = 200 )
    parser.add_argument('--stepsize', '-s', type =float, default = 0.0000007 )
    parser.add_argument('--seed', '-seed', type =int, default = 3)

    #normMode:'Max','Norm','Laplace'
    parser.add_argument('--normMode', '-m', type = str, default = 'Laplace'  )
    parser.add_argument('--coefficient', '-coe', type = float, default = 1 )
    #'ridge','lasso'
    parser.add_argument('--penaMode', '-pena', type = str, default = 'ridge' )


    args = parser.parse_args()

    main(args)
    os.system('say "ロジスティックモデル学習終了"')
    os.system('open -a Finder %s'%datapath)
