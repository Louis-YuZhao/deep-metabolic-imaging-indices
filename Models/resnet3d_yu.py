"""
code for paper titled "Identifying a deep metabolic imaging biomarker to support computer-aided differential diagnosis of Parkinsonism using artificial intelligence"
finished by Yu Zhao 
University of Bern
Technical University of Munich
last modified 07.21.2020
"""
import sys
sys.path.append("..")
from config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu_num']

from keras.layers import Input, Add, Activation 
from keras.layers import Conv3D, UpSampling3D,SpatialDropout3D
from keras.layers import LeakyReLU, BatchNormalization
from keras.layers import MaxPooling3D, GlobalMaxPooling3D, Flatten, Dense
from keras.engine import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_contrib.layers.normalization.groupnormalization import GroupNormalization
# GroupNormalization(groups=2, axis=-1, epsilon=0.1)
from Models.metrics import weighted_dice_coefficient_loss
# from keras import backend as K
# K.image_data_format() == "channels_last"

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']

def resnet3d_model(input_shape=(dimz, dimx, dimy, channelNum), num_outputs=4, 
                    n_base_filters=16, depth=5, dropout_rate=0.3,
                    optimizer=Adam, initial_learning_rate=5e-4,loss_function=weighted_dice_coefficient_loss,
                    kernel_reg_factor=1e-4, ifbase=False):
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters, kernel=(5, 5, 5), strides=(2, 2, 2))
            # in_conv = create_convolution_block(in_conv, n_level_filters, kernel=(5, 5, 5), strides=(2, 2, 2))
            in_conv = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(in_conv)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, kernel=(3, 3, 3), strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)
        summation_layer = Add()([in_conv, context_output_layer])
        
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    output_layer = create_convolution_block(summation_layer, n_level_filters, kernel=(1, 1, 1))
    pool1 = GlobalMaxPooling3D(data_format='channels_last')(output_layer)

    if ifbase == True:
        model = Model(inputs=inputs, outputs=pool1)
        return model
    else: 
        flatten1 = Flatten()(pool1)
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                        kernel_initializer="he_normal",
                        activation="softmax",
                        kernel_regularizer=l2(kernel_reg_factor))(flatten1)
        else:
            dense = Dense(units=num_outputs,
                        kernel_initializer="he_normal",
                        activation="sigmoid",
                        kernel_regularizer=l2(kernel_reg_factor))(flatten1)
        model = Model(inputs=inputs, outputs=dense)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model

def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), normMethod = 'instance_norm'):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if normMethod == 'batch_norm':
        layer = BatchNormalization(axis=-1)(layer)
    elif normMethod == 'instance_norm':
        layer = InstanceNormalization(axis=-1)(layer)
    elif normMethod == 'group_norm':
        layer = GroupNormalization(groups=4, axis=-1, epsilon=0.1)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters,kernel=(3, 3, 3))
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, kernel=(3, 3, 3))
    return convolution2
