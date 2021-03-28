"""
code for paper titled "Differential diagnosis of parkinsonism with deep metabolic imaging biomarker – an artificial intelligence-aided multi-center FDG PET study"
finished by Yu Zhao 
University of Bern
Technical University of Munich
last modified 07.21.2020
"""
import numpy as np
import os
from shutil import copyfile
from config import config
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_num']
from Models import Resnet3DBuilder, resnet3d_model, densenet3d_model
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.layers import Activation, Dense, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from evaluate import evaluate
from keras import backend as K
from skimage.transform import resize
K.image_data_format() == "channels_last"

# Resnet
#%%
# ---------------------------------
# step = 'Step1_Pretraining_2'
# num_outputs = 38
# class_weight = None
# ---------------------------------

# # ---------------------------------
# step = 'Step2'
# # ---------------------------------

# # ---------------------------------
step = 'Step3'
foldNum = 5
# # ---------------------------------
num_outputs = 3

learning_rate = 1e-3

basemodelTrain = True
IFUseWeight = 1 # 1 basemodel, 2 wholemodel, 0 none

baseModeType = 'resnew' #'resori', 'resnew'
epochs = 200
batch_size = 30

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']
#%%
def get_model(baseModeType, reg_factor=1e-4):
    
    if baseModeType == 'resori':
        BaseModel = Resnet3DBuilder.build_resnet_18((dimz, dimx, dimy, channelNum), num_outputs, ifbase=True)
    elif baseModeType == 'resnew':
        BaseModel = resnet3d_model(input_shape=(dimz, dimx, dimy, channelNum), num_outputs=num_outputs, 
                    n_base_filters=64, depth=3, dropout_rate=0.3, ifbase=True)   
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  
    model = Sequential()
    model.add(BaseModel)     
    model.add(Dense(units=num_outputs, kernel_initializer="he_normal", 
            activation="softmax", kernel_regularizer=l2(reg_factor)))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return BaseModel, model

# train
def preprocess(imgs,dimz,dimx,dimy,channelNum):
    imgs_p = np.ndarray((imgs.shape[0], dimz, dimx, dimy, channelNum), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i,:,:,:,0] = resize(imgs[i,:,:,:,0], (dimz, dimx, dimy), preserve_range=True)
    return imgs_p 

def train_and_predict(tempStore):
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30) 
        
    X_train = np.load(os.path.join(tempStore, 'x_train_' + step + '.npy'))
    y_train = np.load(os.path.join(tempStore, 'y_train_' + step + '.npy'))
    X_train = preprocess(X_train,dimz,dimx,dimy,channelNum)
    X_train = X_train.astype('float32')
    # convert class vectors to binary class matrices
    nb_classes = len(np.unique(y_train))
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #---------------------------------#
    BaseModel, model = get_model(baseModeType, reg_factor=1e-4)
    #---------------------------------#
    BaseWeightDir = os.path.join(tempStore, 'BaseWeights.h5')
    weightDir = os.path.join(tempStore, 'Weights.h5')
   
    model_checkpoint = ModelCheckpoint(weightDir, monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    if os.path.exists(BaseWeightDir) and IFUseWeight == 1:
        BaseModel.load_weights(BaseWeightDir)
        if basemodelTrain == False:
            BaseModel.trainable = False
    if os.path.exists(weightDir) and IFUseWeight == 2:
        model.load_weights(weightDir)
    train_history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,\
            verbose=1, shuffle=True,validation_split=0.2, callbacks=[model_checkpoint, early_stop])
    BaseModel.save_weights(BaseWeightDir)

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    np.save(os.path.join(tempStore,'loss.npy'),loss)
    np.save(os.path.join(tempStore,'val_loss.npy'),val_loss)
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    X_test = np.load(os.path.join(tempStore, 'x_test_' + step + '.npy'))
    X_test = preprocess(X_test,dimz,dimx,dimy,channelNum)
    X_test = X_test.astype('float32')
    y_test = np.load(os.path.join(tempStore, 'y_test_' + step + '.npy'))
    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y_test, nb_classes)    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
   
    model.load_weights(weightDir)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    Y_predict = model.predict(X_test, verbose=1)
    np.save(os.path.join(tempStore,'Y_predict.npy'), Y_predict) 

def value_predict(X_test, baseModeType, load_weight_dir, outputDir=None):   
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)  

    X_test = preprocess(X_test,dimz,dimx,dimy,channelNum)
    X_test = X_test.astype('float32')
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    #---------------------------------#
    BaseModel, model = get_model(baseModeType, reg_factor=1e-4)
    #---------------------------------#
    
    model.load_weights(load_weight_dir)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    Y_predict = model.predict(X_test, verbose=1)
    if outputDir != None:
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
            np.save(os.path.join(outputDir,'Y_predict.npy'), Y_predict)
    return Y_predict 

if __name__ == '__main__':
    currentOpePath = os.path.realpath(__file__)
    print(currentOpePath)
    currBname = os.path.basename(currentOpePath) 
    if step == 'Step3':
        tempStoreRoot = './tempData'
        tempStore = os.path.join(tempStoreRoot, 'fold_'+str(foldNum))
        train_and_predict(tempStore)
        copyfile(currentOpePath, os.path.join(tempStore,'train.py'))
    else:
        tempStoreRoot = './tempData'
        tempStore = os.path.join(tempStoreRoot, 'pretrain')
        train_and_predict(tempStore)
        copyfile(currentOpePath, os.path.join(tempStore,'train.py'))
    if learning_rate == 5e-4:
        evaluate(foldNum)
