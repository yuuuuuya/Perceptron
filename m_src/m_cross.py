import numpy as np
import pdb#pdb.set_trace()
import os.path
import sys
sys.path.append('/Users/inagakiyuuya/Dropbox/Inagaki/dataset/mnist_data')

import m_datautil as m_data

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold


'''
split inputdata and outputdta for cross validataion.
inputdata and outputdata are orign data.
input  : inputdata(60000,784), outputdata(60000,1), splits(splits number), norm modele
output :
train_record
first dats is first cross validation data. the data normalized by  norm modele.
...
i-th          i-th

test_record
same explain with train_record.
'''
def CroVali_Data(inputdata, outputdata, splits, normMode = 'Laplace'):
    n_splits =  splits
    epochIdx = 0
    train_record = []
    test_record = []
    #cross validation(split array and loop)
    for train_idx, test_idx in StratifiedKFold(n_splits).split(inputdata, outputdata):
        xs_train = inputdata[train_idx]
        t_train = outputdata[train_idx]
        xs_test = inputdata[test_idx]
        t_test = outputdata[test_idx]
        #matched number of occupancy=0 with occupancy=1
        match_train_input, match_train_output = m_data.dataMatch(xs_train, t_train, classNum = 10)
        norm_train_input = m_data.Normalization(match_train_input, normMode)
        norm_test_input = m_data.Normalization(xs_test, normMode)

        traindatas = [norm_train_input, match_train_output]
        testdatas = [norm_test_input, t_test]

        #Seving to a set matchX and matchT
        train_record = train_record + [traindatas]
        #Seving to a set xs_test and T_test
        test_record = test_record +[testdatas]

        epochIdx+=1

    return train_record, test_record
