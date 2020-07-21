#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 09:54:49 2019

@author: zhaoyu
"""

import os
import glob
import string
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk

#%%
#--------------------------------------
step = 'Step2'
testNumPer = 0.1
#--------------------------------------
#step = 'Step3'
#kfold = 6
#SEED = 448
#--------------------------------------
tempStoreRoot = './tempData'
IfNorm = True

dataDir = os.path.join('../PD_PET_Data',step)
Format1 = '*.hdr'
fileList1 = glob.glob((dataDir+'/*/'+Format1))
fileList1.sort()
Format2 = '*.img'
fileList2 = glob.glob((dataDir+'/*/'+Format2))
fileList2.sort()

#%%
def splitData(X_data,Y_data, testNumPer):
    SampleNum = np.shape(X_data)[0]
    currentList = range(SampleNum)
    random.shuffle(currentList)
    testNum = int(testNumPer*SampleNum)
    testID = currentList[:testNum]
    trainingID = currentList[testNum:]
    X_train = X_data[trainingID,:]
    Y_train = Y_data[trainingID,:]
    X_test = X_data[testID,:]
    Y_test = Y_data[testID,:]
    return X_train, Y_train, X_test, Y_test

def splitData_Kfold(X_data,Y_data,kfold,SEED):

    dataFoldList = list()
    
    SEED = SEED
    SampleNum = np.shape(X_data)[0]
    foldSize = int(SampleNum/kfold)
    currentList = range(SampleNum)
    random.seed(SEED)
    random.shuffle(currentList)
    
    for i in range(kfold):
        DataDict = dict()

        if i != kfold:
            testID = currentList[(i*foldSize):(i*foldSize+foldSize)]
            trainingID = list(set(currentList) - set(testID))
        else:
            testID = currentList[(kfold*foldSize):]
            trainingID = list(set(currentList) - set(testID))
        
        random.shuffle(testID)
        random.shuffle(trainingID)
        DataDict['X_train'] = X_data[trainingID,:]
        DataDict['Y_train'] = Y_data[trainingID,:]
        DataDict['X_test'] = X_data[testID,:]
        DataDict['Y_test'] = Y_data[testID,:]

        dataFoldList.append(DataDict) 
        
    return dataFoldList

def dataPrepare(fileList1, fileList2, IfNorm):
    arrayDict = dict()
    labelDict = dict()
    for i in range(len(fileList1)):
        # chage file name of .img
        currentfile = fileList1[i]    
        name, ext = os.path.splitext(currentfile)
        Bname = os.path.basename(currentfile)
        Dname = os.path.dirname(currentfile)
        item = string.join(Bname.split("_")[0:2], "_")
        item = string.join(item.split(".")[0:1], ".")
        newname = os.path.join(Dname,item+ext)
        os.rename(currentfile, newname)

        # chage file name of .hdr
        currentfile = fileList2[i]    
        name, ext = os.path.splitext(currentfile)
        Bname = os.path.basename(currentfile)
        Dname = os.path.dirname(currentfile)
        item = string.join(Bname.split("_")[0:2], "_")
        item = string.join(item.split(".")[0:1], ".")
        newname = os.path.join(Dname,item+ext)
        os.rename(currentfile, newname)

        currentImg = sitk.ReadImage(newname)
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

        if step+'_MSA' in Dname:
            labelDict[item]=np.int64(0)
        elif step+'_PD' in Dname:
            labelDict[item]=np.int64(1)
        elif step+'_PSP' in Dname:
            labelDict[item]=np.int64(2)
        else:
            raise ValueError('the item is not correctly classifed') 
        
    new_labelDict = sorted(labelDict.items(), key = lambda labelDict:labelDict[0], reverse=False)

    # write labels to csv file
    list_key = []
    list_value = []
    for i in range(len(new_labelDict)):
        list_key.append(new_labelDict[i][0])
        list_value.append(new_labelDict[i][1])
    dataRestore = {'patientID':list_key, 'patientValue':list_value}
    df = pd.DataFrame(data=dataRestore)
    outCSV = os.path.join(dataDir,'label.csv')
    df.to_csv(outCSV)    

    # prepare trainging data and label
    sorted_keys = sorted(labelDict.keys(), reverse=False)
    NN = len(sorted_keys)
    ImgArray = np.zeros((NN,shapeZ,shapeX,shapeY,1))
    LabelArray = np.zeros((NN,1))

    ii=0
    for currentKey in sorted_keys:
        ImgArray[ii,:,:,:,0] =  arrayDict[currentKey]
        LabelArray[ii,:] = labelDict[currentKey]
        ii += 1    
    return ImgArray,LabelArray

if __name__ == '__main__':
    ImgArray,LabelArray = dataPrepare(fileList1, fileList2, IfNorm)    
    if not os.path.exists(tempStoreRoot):
        os.mkdir(tempStoreRoot)

    if step == 'Step2':
    
        X_train, Y_train, X_test, Y_test = splitData(ImgArray, LabelArray, testNumPer)
        tempStore = os.path.join(tempStoreRoot, 'pretrain')
        if not os.path.exists(tempStore):
            os.mkdir(tempStore)
        np.save(os.path.join(tempStore, 'x_train_' + step + '.npy'), X_train)
        np.save(os.path.join(tempStore, 'y_train_' + step + '.npy'), Y_train)
        np.save(os.path.join(tempStore, 'x_test_' + step + '.npy'), X_test)
        np.save(os.path.join(tempStore, 'y_test_' + step + '.npy'), Y_test)
    
    elif step == 'Step3':           
    
        dataFoldList = splitData_Kfold(ImgArray,LabelArray,kfold,SEED)
        for foldNum in range(len(dataFoldList)):
            tempStore = os.path.join(tempStoreRoot, 'fold_'+str(foldNum))
            if not os.path.exists(tempStore):
                os.mkdir(tempStore)
            np.save(os.path.join(tempStore, 'x_train_' + step + '.npy'), (dataFoldList[foldNum])['X_train'])
            np.save(os.path.join(tempStore, 'y_train_' + step + '.npy'), (dataFoldList[foldNum])['Y_train'])
            np.save(os.path.join(tempStore, 'x_test_' + step + '.npy'), (dataFoldList[foldNum])['X_test'])
            np.save(os.path.join(tempStore, 'y_test_' + step + '.npy'), (dataFoldList[foldNum])['Y_test'])
