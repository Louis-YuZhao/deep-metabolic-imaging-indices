#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jan 20 2019

@author: zhaoyu
"""

import glob
import os
import string
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from config import config
from train import value_predict

#%%
disease_names = ['MSA', 'PD', 'PSP']

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']
#%%
def reWriteFiles(imgFileList, IfNorm):
    arrayDict = dict()
    for i in range(len(imgFileList)):
        # chage file name of .img
        currentfile = imgFileList[i]    
        name, ext = os.path.splitext(currentfile)
        Bname = os.path.basename(name)
        if Bname.startswith('swDP_'):
            item = "_".join(Bname.split("_")[0:2])
        else:
            item = Bname
        currentImg = sitk.ReadImage(currentfile)
        currentArray = sitk.GetArrayFromImage(currentImg)
        if IfNorm == True:
            mean = np.mean(currentArray)  # mean for data centering
            std = np.std(currentArray)  # std for data normalization
            currentArray -= mean
            currentArray /= std
        arrayDict[item] = currentArray
        if i == 0:
            shapeZ = currentArray.shape[0]
            shapeX = currentArray.shape[1]
            shapeY = currentArray.shape[2]
    return arrayDict, shapeZ, shapeX, shapeY

def WritePredictResultToXLSX(list_ID, Y_predict, Y_predict_name, Y_possibility, outXLSX):
    dataRestore = {'patientID':list_ID, 'patientValue':Y_predict_name, 'Prediction':Y_predict, 'PosMSA':Y_possibility[:,0], 'PosPID':Y_possibility[:,1], 'PosPSP':Y_possibility[:,2]}
    df = pd.DataFrame(data=dataRestore)    
    df.to_excel(outXLSX, index=None,encoding='utf_8_sig', columns=['patientID', 'patientValue', 'Prediction', 'PosMSA', 'PosPID', 'PosPSP'])    

def finalTestDataPrepare(imgFileList,
                         tempDataStore, 
                         IfNorm, 
                         IfSave = True):
    arrayDict, shapeZ, shapeX, shapeY = reWriteFiles(imgFileList, IfNorm)

    # prepare trainging data and label
    sorted_keys = sorted(arrayDict.keys(), reverse=False)
    NN = len(sorted_keys)
    ImgArray = np.zeros((NN,shapeZ,shapeX,shapeY))

    ii=0
    for currentKey in sorted_keys:
        ImgArray[ii,:,:,:] =  arrayDict[currentKey]
        ii += 1

    if IfSave == True:
        if not os.path.exists(tempDataStore):
            os.mkdir(tempDataStore)
        np.save(os.path.join(tempDataStore, 'x_train' + '.npy'), ImgArray)
        np.save(os.path.join(tempDataStore, 'ID_train' + '.npy'), sorted_keys)
    
    return ImgArray, sorted_keys

def oneModelEvaluate(ImgArray, list_ID, BMtype, load_weight_dir, disease_names, outXLSX):    
    y_predict = value_predict(ImgArray, BMtype, load_weight_dir, outputDir=None)
    Y_predict = np.argmax(y_predict, axis=1)
    Y_predict_name = list()
    for item in Y_predict:
        Y_predict_name.append(disease_names[item])
    WritePredictResultToXLSX(list_ID, Y_predict_name, outXLSX)

def MultiModelEvaluate(ImgArray, list_ID, BM_list, weight_dir_list, disease_names, outXLSX):
    y_predict_list = list()
    for i in range(len(weight_dir_list)):
        y_predict = value_predict(ImgArray, BM_list[i], weight_dir_list[i], None)
        y_predict_list.append(y_predict)
        print(i)
    y_predict_mean = np.mean(np.array(y_predict_list), axis=0)
    Y_possibility = y_predict_mean
    Y_predict = np.argmax(Y_possibility, axis=1)
    Y_predict_name = list()
    for item in Y_predict:
        Y_predict_name.append(disease_names[item])
    WritePredictResultToXLSX(list_ID, Y_predict, Y_predict_name, Y_possibility, outXLSX)

def main():
    parser = argparse.ArgumentParser(description = "PDDNET command line tool")
    parser.add_argument("--data_folder", type=str, help = "Input directory path")
    parser.add_argument("--project_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--image_ext", type=str, help = "image extension.", default = '*.img')
    parser.add_argument("--data_norm", help = "Whether conduct data normalization.", action = 'store_true')
    parser.add_argument("--baseModeType", type=str, default='resnew', help = "network type: 'resori', 'resnew', 'dense'")
    parser.add_argument('--CV_fold', type=int, default=6, help='cross validation fold')
    args = parser.parse_args()

    #%%
    # original data
    if not os.path.exists(args.project_folder):
        os.makedirs(args.project_folder)        

    imgFileList = glob.glob((args.data_folder + args.image_ext))
    imgFileList.sort()

    ImgArray, list_ID = finalTestDataPrepare(imgFileList,
                                             args.project_folder, 
                                             IfNorm = args.data_norm, 
                                             IfSave = True)
    # resnew 1_1_1
    model_list_resnew = list()
    resnew_list = [args.baseModeType]*6
    for currentFold in range(1, args.CV_fold+1):
        model_list_resnew.append(os.path.join(args.project_folder,'data_{0}_{1}'.format(args.CV_fold, currentFold), 'Weights.h5'))

    outXLSX = os.path.join(args.project_folder, 'Test_result_model_resnew.xlsx')
    MultiModelEvaluate(ImgArray, list_ID, resnew_list, model_list_resnew, disease_names, outXLSX)

if __name__ == '__main__':
    main()