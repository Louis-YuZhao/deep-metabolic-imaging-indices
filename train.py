import os

from matplotlib.colors import from_levels_and_colors
import numpy as np
import argparse
from config import config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_num']
from Models import Resnet3DBuilder, resnet3d_model, densenet3d_model
from Models.training import get_callbacks
from Models.generator import convertData, preprocess, DataGeneratorNew
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.layers import Activation, Dense, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from evaluate import evaluate
from keras import backend as K
from skimage.transform import resize
from math import ceil
K.image_data_format() == "channels_last"

#%%
class_weight = None
dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']
#%%
def get_model_new(baseModeType, num_outputs, optType, learning_rate, reg_factor=1e-4):
    
    if baseModeType == 'resori':
        model = Resnet3DBuilder.build_resnet_18((dimz, dimx, dimy, channelNum), num_outputs, 
                                                reg_factor = reg_factor, ifbase=False)
    elif baseModeType == 'resnew':
        model = resnet3d_model(input_shape=(dimz, dimx, dimy, channelNum), num_outputs=num_outputs, 
                    n_base_filters=64, depth=3, dropout_rate=0.3, kernel_reg_factor = reg_factor, ifbase=False)
    elif baseModeType == 'dense':
        model = densenet3d_model(input_shape=(dimz, dimx, dimy, channelNum), num_outputs=num_outputs, 
                    n_base_filters=64, depth=3, dropout_rate=0.3, kernel_reg_factor = reg_factor, ifbase=False)     
    # Name layers
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
    
    if optType == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optType == 'sgd':
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)        
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model

# train
def train_and_predict(projectRoot, args):
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30) 

    inputDict = dict()
    inputDict['MSA'] = os.path.join(projectRoot, 'MSA.npy')
    inputDict['PID'] = os.path.join(projectRoot, 'PID.npy')
    inputDict['PSP'] = os.path.join(projectRoot, 'PSP.npy')

    features=dict()
    features['MSA'] = np.load(inputDict['MSA'])
    features['PID'] = np.load(inputDict['PID'])
    features['PSP'] = np.load(inputDict['PSP'])
    
    itemNum = int(0)
    for key, value in features.items():
        itemNum = itemNum+value.shape[0]
    steps_per_epoch = ceil(itemNum / args.batch_size) 

    labelDef = config['labelDef']
   
    labelPercentage = dict()
    labelPercentage['MSA'] = 1/float(3)
    labelPercentage['PSP'] = 1/float(3)
    labelPercentage['PID'] = 1/float(3)

    X_test = np.load(os.path.join(projectRoot,'testFeatures.npy'))
    y_test = np.load(os.path.join(projectRoot,'testLabels.npy'))
    X_test,Y_test = convertData(X_test,y_test, config) 
    validation_data = (X_test,Y_test) 
    TrainData = DataGeneratorNew(features, labelDef, labelPercentage, args.batch_size)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #---------------------------------#
    model = get_model_new(args.baseModeType, args.num_outputs, args.optType, args.learning_rate, reg_factor=1e-4)
    #---------------------------------#
    weightDir = os.path.join(projectRoot, 'Weights.h5')
   
    callbackList = get_callbacks(weightDir, args.learning_rate, args.learning_rate_drop, args.learning_rate_patience, 
                            learning_rate_epochs=None, logging_file="training.log", verbosity=1,
                            early_stopping_patience=args.early_stopping_patience)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)   

    if os.path.exists(args.pretrained_weight_dir) and args.UseWeight == True:
        model.load_weights(args.pretrained_weight_dir)
    if class_weight != None:
        train_history = model.fit_generator(TrainData.generator(), 
                                            steps_per_epoch=steps_per_epoch, 
                                            epochs=args.epochs, 
                                            verbose=1, 
                                            callbacks=callbackList, 
                                            validation_data=validation_data, 
                                            validation_steps=1, 
                                            class_weight=class_weight, 
                                            max_queue_size=10, 
                                            workers=1, 
                                            use_multiprocessing=False)

    else:
        train_history = model.fit_generator(TrainData.generator(), 
                                            steps_per_epoch=steps_per_epoch, 
                                            epochs=args.epochs, 
                                            verbose=1, 
                                            callbacks=callbackList, 
                                            validation_data=validation_data, 
                                            validation_steps=1, 
                                            max_queue_size=10, 
                                            workers=1, 
                                            use_multiprocessing=False)

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    np.save(os.path.join(projectRoot,'loss.npy'),loss)
    np.save(os.path.join(projectRoot,'val_loss.npy'),val_loss)
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    X_test = np.load(os.path.join(projectRoot,'testFeatures.npy'))
    y_test = np.load(os.path.join(projectRoot,'testLabels.npy'))
    X_test, Y_test = convertData(X_test,y_test, config)    
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
    np.save(os.path.join(projectRoot,'Y_predict.npy'), Y_predict) 

def value_predict(X_test, baseModeType, optType, load_weight_dir, outputDir=None):   
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)  

    X_test = preprocess(X_test,dimz,dimx,dimy,channelNum)
    X_test = X_test.astype('float32')
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    #---------------------------------#
    model = get_model_new(baseModeType, optType, reg_factor=1e-4)
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

def main():
    parser = argparse.ArgumentParser(description = "PDDNET command line tool")
    parser.add_argument("--project_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--baseModeType", type=str, default='resnew', help = "network type: 'resori', 'resnew', 'dense'")
    parser.add_argument("--num_outputs", type=int, default=3, help = "class number")
    parser.add_argument("--optType", type=str, default='adam', help = "optimizer type: 'adam', 'sgd'")
    parser.add_argument("--epochs", type=int, default=30, help = "training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help = "class number")
    parser.add_argument('--CV_fold', type=int, default=6, help='cross validation fold')
    parser.add_argument('--currentFold', type=int, default=1, help='current training fold')
    parser.add_argument("--UseWeight", help = "Whether conduct data normalization.", action = 'store_true')
    parser.add_argument("--pretrained_weight_dir", type=str, default=None, help = "pretrained weights as the starting point, usefull if UseWeight=True")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--learning_rate_drop', type=float, default=0.5, help='learning rate drop')
    parser.add_argument('--learning_rate_patience', type=int, default=10, help='learning rate drop patience')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()

    currentOpePath = os.path.realpath(__file__)
    print(currentOpePath)
    projectRoot = os.path.join(args.project_folder, 'data_{0}_{1}'.format(kfold=args.CV_fold, currentFold=args.currentFold))
    train_and_predict(projectRoot, args)
    evaluate(projectRoot)

if __name__ == '__main__':
    main()