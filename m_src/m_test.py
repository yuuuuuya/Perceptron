import numpy as np
import pdb
import sys
sys.path.append('/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data')
import os.path
import argparse

import m_predictutil as pre
import LoadMNIST as load
import m_datautil as m_data

parameterpath = '../result_folder/'
m_datapath = '/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data'
imgTest = load.readMNIST('train_img', m_datapath)
labTest = load.readMNIST('train_label', m_datapath)

def para_load(parameterpath, filename = 'para_cross_step7e-07_iter200_ridge_lamb1.0.npy'):
    filepath = os.path.join(parameterpath, filename )
    sys.path.append('parameterpath')
    load_data = np.load(filepath)
    paraW = load_data[()]

    return paraW



def main(args):

    #test data
    imgTest = load.readMNIST('train_img', m_datapath)
    labTest = load.readMNIST('train_label', m_datapath)

    normTest = m_data.Normalization(imgTest, mode = args.normMode)


    #train_paraW
    paraW = para_load(parameterpath, filename = args.parafile)

    normTest = m_data.Normalization(imgTest, mode = args.normMode)

    #predicte
    predictVec = pre.predict_array(normTest, paraW, args.coeffi, args.penaMode)

    #compute
    accu = pre.accuracy(predictVec, labTest)

    print(accu)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Test')

    parser.add_argument('--parafile', '-pf', type = str, default = 'para_cross_step7e-07_iter200_ridge_lamb1.0.npy')
    parser.add_argument('--coeffi', '-coe', type = int, default = 1  )
    #normMode:'Max','Norm','Laplace'
    parser.add_argument('--normMode', '-m', type = str, default = 'Laplace'  )
    parser.add_argument('--penaMode', '-pena', type = str, default = 'ridge', help = 'penalty model(regularization)')


    args = parser.parse_args()

    main(args)
    os.system('say "テスト終了"')
