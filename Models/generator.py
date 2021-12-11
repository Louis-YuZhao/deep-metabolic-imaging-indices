# -*- coding: utf-8 -*-
"""
Data generator

By Yu Zhao, 2019.04.15
"""
import sys
import os
import random
import numpy as np
sys.path.append("..")
from config import config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu_num']
from skimage.transform import resize

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']
#%%

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.   
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def preprocess(imgs, dimz, dimx, dimy, channelNum):
    imgs_p = np.ndarray((imgs.shape[0], dimz, dimx, dimy, channelNum), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i,:,:,:,0] = resize(imgs[i,:,:,:], (dimz, dimx, dimy), preserve_range=True)
    return imgs_p 

def convertData(image, label, config):
    dimz = config['dimz']
    dimx = config['dimx']
    dimy = config['dimy']
    channelNum = config['channelNum']
    X = preprocess(image, dimz, dimx, dimy, channelNum)
    X = X.astype('float32')
    # convert class vectors to binary class matrices
    nb_classes = len(np.unique(label))
    Y = to_categorical(label, nb_classes)
    return X, Y

class DataGeneratorNew():
    def __init__(self, features, labels, labelPercentage, batch_size):
        self.features = features
        self.labels = labels
        self.labelPercentage = labelPercentage
        self.batch_size = batch_size
        self.batch_train_features = None 
        self.batch_train_labels = None

    def __data_generator(self, feature, label, batch_size):

        F_N = np.shape(feature)[0]
        # choose random index in features
        index= random.sample(range(F_N),batch_size)
        batch_features = feature[index]
        batch_labels = label[index]

        return batch_features, batch_labels

    def generator(self,):
        # Create empty arrays to contain batch of features and labels
        while True:
            batch_features_list = []
            batch_labels_list = []
            for key in self.features.keys():
                feature = self.features[key]
                label = self.labels[key] * np.ones(feature.shape[0],dtype=int)
                subBatchSize = int(self.labelPercentage[key]*self.batch_size)
                batch_feature, batch_label = self.__data_generator(feature, label, subBatchSize)
                batch_features_list.append(batch_feature)
                batch_labels_list.append(batch_label)
            batch_train_features, batch_train_labels = self.__batch_Data(batch_features_list, batch_labels_list)
            self.batch_train_features, self.batch_train_labels = convertData(batch_train_features, batch_train_labels, config)
            yield self.batch_train_features, self.batch_train_labels

    def __batch_Data(self, batch_features_list, batch_labels_list):
            batch_features = np.concatenate(batch_features_list,axis=0)
            batch_labels = np.concatenate(batch_labels_list,axis=0)
            Num = batch_features.shape[0]
            index = random.sample(range(Num),Num)         
            batch_features = batch_features[index]
            batch_labels = batch_labels[index]
            return batch_features, batch_labels   

    
class DataGenerator():
    def __init__(self, features, labels, labelPercentage, batch_size, valid_size):
        self.features = features
        self.labels = labels
        self.labelPercentage = labelPercentage
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.batch_train_features, self.batch_train_labels, self.batch_valid_features, self.batch_valid_labels = \
            next(self.__generator_train_validation())

    def __data_generator(self, feature, label, batch_size):

        F_N = np.shape(feature)[0]
        # choose random index in features
        index= random.sample(range(F_N),batch_size)
        batch_features = feature[index]
        batch_labels = label[index]

        return batch_features, batch_labels

    def __generator(self,):
        # Create empty arrays to contain batch of features and labels
        while True:
            batch_features_list = []
            batch_labels_list = []
            for key in self.features.keys():
                feature = self.features[key]
                label = self.labels[key] * np.ones(feature.shape[0],dtype=int)
                subBatchSize = int(self.labelPercentage[key]*self.batch_size)
                batch_feature, batch_label = self.__data_generator(feature, label, subBatchSize)
                batch_features_list.append(batch_feature)
                batch_labels_list.append(batch_label)
            batch_features = np.array(batch_features_list)
            batch_labels = np.array(batch_labels_list)
            index = random.sample(range(self.batch_size), self.batch_size)
            batch_features = batch_features[index]
            batch_labels = batch_labels[index]
            yield batch_features, batch_features

    def __generator_train_validation(self,):
        # Create empty arrays to contain batch of features and labels
        while True:
            batch_features_list_train = []
            batch_labels_list_train = []
            batch_features_list_valid = []
            batch_labels_list_valid = []
            for key in self.features.keys():
                feature = self.features[key]
                label = self.labels[key] * np.ones(feature.shape[0],dtype=int)
                subBatchSize = int(self.labelPercentage[key]*self.batch_size)
                subValidSize = int(self.labelPercentage[key]*self.valid_size)
                batch_feature, batch_label = self.__data_generator(feature, label, (subBatchSize+subValidSize))
                
                batch_features_list_train.append(batch_feature[0:subBatchSize])
                batch_labels_list_train.append(batch_label[0:subBatchSize])
                
                batch_features_list_valid.append(batch_feature[subBatchSize: subBatchSize+subValidSize])
                batch_labels_list_valid.append(batch_label[subBatchSize: subBatchSize+subValidSize])
                
            batch_train_features, batch_train_labels = self.__batch_Data(batch_features_list_train, batch_labels_list_train)
            batch_valid_features, batch_valid_labels = self.__batch_Data(batch_features_list_valid, batch_labels_list_valid)

            yield batch_train_features, batch_train_labels, batch_valid_features, batch_valid_labels

    def __batch_Data(self, batch_features_list, batch_labels_list):
            batch_features = np.concatenate(batch_features_list,axis=0)
            batch_labels = np.concatenate(batch_labels_list,axis=0)
            Num = batch_features.shape[0]
            index = random.sample(range(Num),Num)         
            batch_features = batch_features[index]
            batch_labels = batch_labels[index]
            return batch_features, batch_labels   
    
    def training_generator(self,):
        while True:
            self.batch_train_features, self.batch_train_labels, self.batch_valid_features, self.batch_valid_labels = \
            next(self.__generator_train_validation())
            batch_train_features, batch_train_labels = convertData(self.batch_train_features, self.batch_train_labels, config)
            yield batch_train_features, batch_train_labels 

    def valiation_generator(self,):
        while True:
            batch_valid_features, batch_valid_labels = convertData(self.batch_valid_features, self.batch_valid_labels, config)
            yield batch_valid_features, batch_valid_labels

