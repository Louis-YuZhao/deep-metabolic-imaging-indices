#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 09:54:49 2019

@author: zhaoyu
"""
#%%
import os
import glob
import argparse
import numpy as np
import random
from random import shuffle
import pandas as pd
import SimpleITK as sitk 
from config import config

#%% s
def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data

def group_normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage

def singal_normalize_data_storage(currentArray):
    currentArray = np.float64(currentArray)
    mean = np.mean(currentArray)  # mean for data centering
    std = np.std(currentArray)  # std for data normalization
    currentArray -= mean
    currentArray /= std
    return currentArray

#%%
class classificationDataPrepare(object):
    
    def __init__(self, inputDict, labelDef, outputFolder):
        # input data
        self.inputDict = inputDict
        self.labelDef = labelDef
        #         
        self.classList = list(inputDict.keys())
        self.FilesInDiffClassDict = dict()
        self.DataInDiffClassDict = dict()
        # output data
        self.outputFolder = outputFolder
        self.outputTrainDict = dict()
        self.outputTestPID = None
        self.outputTestFeatures = None
        self.outputTestLabels = None
    
    def __GetPID(self,fileDir):
        name, _ = os.path.splitext(fileDir)
        Bname = os.path.basename(name)
        item = "_".join(Bname.split('_')[0:2])
        return item

    def readInput(self, checkItem, randomSeed=None):
        for key in self.classList:
            self.FilesInDiffClassDict[key] = glob.glob(os.path.join(self.inputDict[key],checkItem))
            self.FilesInDiffClassDict[key].sort()
            if randomSeed:
                random.Random(randomSeed).shuffle(self.FilesInDiffClassDict[key])
    
    def GetRawData(self, IfNorm=True):
        for key in self.classList:
            self.DataInDiffClassDict[key] = list()
            itemList = self.FilesInDiffClassDict[key]
            for item in itemList:
                itemInfo = dict()
                itemInfo['PID'] = self.__GetPID(item)
                print(item)
                itemInfo['Array'] = sitk.GetArrayFromImage(sitk.ReadImage(item))
                if IfNorm:
                    itemInfo['Array'] = singal_normalize_data_storage(itemInfo['Array'])
                print(np.shape(itemInfo['Array']))
                itemInfo['Label'] = self.labelDef[key]
                self.DataInDiffClassDict[key].append(itemInfo)

    def Train_Test_rawData(self, testdefDict):
        FinalTestList=[]
        for key in self.classList:
            currTrainList, currTestList = self.__Train_Test_list(self.DataInDiffClassDict[key], testdefDict[key], shuffleList=True)
            FinalTestList = FinalTestList + currTestList
            currTrainDir = os.path.join(self.outputFolder, key+'.npy')
            self.outputTrainDict[key] = currTrainDir
            self.__Transfer_list_to_data(currTrainList, currTrainDir)
        
        shuffle(FinalTestList)
        self.outputTestPID = os.path.join(self.outputFolder, 'testInformation.csv')
        self.outputTestFeatures = os.path.join(self.outputFolder, 'testFeatures.npy')
        self.outputTestLabels = os.path.join(self.outputFolder, 'testLabels.npy')
        self.__Transfer_list_to_data(FinalTestList, self.outputTestFeatures, self.outputTestLabels)
        return self.outputTrainDict, self.outputTestFeatures, self.outputTestLabels
    
    def Train_Test_rawData_Kfold(self, kfold, currentFold):
        FinalTestList=[]
        for key in self.classList:
            currTrainList, currTestList = self.__Train_Test_list_Kfold(self.DataInDiffClassDict[key], kfold, currentFold, shuffleList=False)
            FinalTestList = FinalTestList + currTestList
            currTrainDir = os.path.join(self.outputFolder, key+'.npy')
            self.outputTrainDict[key] = currTrainDir
            self.__Transfer_list_to_data(currTrainList, saveFeatures = currTrainDir)
        
        shuffle(FinalTestList)
        self.outputTestPID = os.path.join(self.outputFolder, 'testInformation.csv')
        self.outputTestFeatures = os.path.join(self.outputFolder, 'testFeatures.npy')
        self.outputTestLabels = os.path.join(self.outputFolder, 'testLabels.npy')
        self.__Transfer_list_to_data(FinalTestList, self.outputTestFeatures, self.outputTestLabels)
        return self.outputTrainDict, self.outputTestFeatures, self.outputTestLabels
    
    def __Train_Test_list(self, dataList, testNum, shuffleList=False):
        Num = len(dataList)
        TrainNum = int(Num - testNum)
        order = range(Num)

        if shuffleList:
            shuffle(order)

        train_order = order[0:TrainNum]
        test_order = order[TrainNum:Num]
        return dataList[train_order], dataList[test_order]

    def __Train_Test_list_Kfold(self, dataList, kfold, currentFold, shuffleList=False):   

        SampleNum = len(dataList)
        foldSize = int(SampleNum/kfold)
        currentList = range(SampleNum)          

        if currentFold != kfold:
            currentFold = currentFold-1
            testID = currentList[(currentFold*foldSize):(currentFold*foldSize+foldSize)]
            trainingID = list(set(currentList) - set(testID))
        else:
            currentFold = currentFold-1
            testID = currentList[(currentFold*foldSize):]
            trainingID = list(set(currentList) - set(testID))
            
        TrainList=[]
        TestList = []
        if shuffleList:
            random.shuffle(testID)
            random.shuffle(trainingID)
        for index in trainingID:
            TrainList.append(dataList[index])
        for index in testID:
            TestList.append(dataList[index])           
        return TrainList, TestList
    
    def __Transfer_list_to_data(self, dataList, saveFeatures = None, saveLabels = None):
        outputListPID = []
        outputListFeatures = []
        outputListLabels = []
        for item in dataList:
            outputListPID.append(item['PID'])
            outputListFeatures.append(item['Array'])
            outputListLabels.append(item['Label'])
        # save the test information
        df = pd.DataFrame({'PID': outputListPID, 'Label': outputListLabels})
        df.to_csv(self.outputTestPID, index=None)
        if saveFeatures:
            np.save(saveFeatures, np.array(outputListFeatures))
        if saveLabels:
            np.save(saveLabels, np.array(outputListLabels))
        return np.array(outputListFeatures), np.array(outputListLabels)

def main():
    parser = argparse.ArgumentParser(description = "PDDNET command line tool")
    parser.add_argument("--data_folder", type=str, help = "Input directory path")
    parser.add_argument("--project_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--image_ext", type=str, help = "image extension.", default = '*.img')
    parser.add_argument('--CV_fold', type=int, default=6, help='cross validation fold')
    parser.add_argument('--random_seed', type=int, default=448, help='random seed')
    parser.add_argument("--data_norm", help = "Whether conduct data normalization.", action = 'store_true')
    args = parser.parse_args()
    
    inputDict = dict()
    inputDict['MSA'] = os.path.abspath(os.path.join(args.data_folder, 'MSA'))  
    inputDict['PID'] = os.path.abspath(os.path.join(args.data_folder, 'IPD'))
    inputDict['PSP'] = os.path.abspath(os.path.join(args.data_folder, 'PSP'))

    labelDef = config['labelDef']

    for currentFold in range(1, args.CV_fold+1):
        outputFolder = os.path.abspath(os.path.join(args.project_folder, 'data_{0}_{1}'.format(args.CV_fold,currentFold)))
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        dataPrepare = classificationDataPrepare(inputDict, labelDef, outputFolder)
        dataPrepare.readInput(checkItem=args.image_ext, randomSeed=args.random_seed)
        dataPrepare.GetRawData(IfNorm = args.data_norm)
        dataPrepare.Train_Test_rawData_Kfold(args.CV_fold, currentFold)   
#%%    
if __name__ =='__main__':
    main()