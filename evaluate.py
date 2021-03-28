"""
code for paper titled "Differential diagnosis of parkinsonism with deep metabolic imaging biomarker – an artificial intelligence-aided multi-center FDG PET study"
finished by Yu Zhao 
University of Bern
Technical University of Munich
last modified 07.21.2020
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import classification_report

def evaluate(foldNum):
    step = 'Step3'
    foldNum = foldNum
    tempStoreRoot = './tempData'
    tempStore = os.path.join(tempStoreRoot, 'fold_'+str(foldNum))

    # ground truth
    Y_test = np.load(os.path.join(tempStore, 'y_test_' + step + '.npy'))
    Y_test = np.squeeze(Y_test)
    Y_test= np.int64(Y_test)
    print Y_test
    print Y_test.dtype

    # predict result
    y_predict = np.load(os.path.join(tempStore,'Y_predict.npy'))
    # print y_predict
    Y_predict = np.argmax(y_predict, axis=1)
    print Y_predict
    print Y_predict.dtype

    #classification_report
    target_names = ['MSA', 'PD', 'PSP']
    print(classification_report(Y_test, Y_predict, target_names=target_names))
    with open(os.path.join(tempStore,'evaResult.txt'),mode='w') as file_handle:
        file_handle.write(classification_report(Y_test, Y_predict, target_names=target_names))
    
if __name__ == "__main__":
    evaluate(foldNum)
