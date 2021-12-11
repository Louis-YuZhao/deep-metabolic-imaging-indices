import sys
sys.path.append("..")
from config import config
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu_num']

from keras.layers import Input, Add, Activation 
from keras.layers import Conv3D, UpSampling3D,SpatialDropout3D
from keras.layers import LeakyReLU, BatchNormalization, concatenate
from keras.layers import GlobalMaxPooling3D, GlobalAveragePooling3D, MaxPooling3D, Flatten, Dense
from keras.engine import Model
from keras.optimizers import Adam
from keras.regularizers import l2
# from tensorflow_addons.layers import InstanceNormalization
# from tensorflow_addons.layers import GroupNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_contrib.layers.normalization.groupnormalization import GroupNormalization
# GroupNormalization(groups=2, axis=-1, epsilon=0.1)
from Models.metrics import weighted_dice_coefficient_loss
from keras import backend as K
K.image_data_format() == "channels_last"
data_format=K.image_data_format()
channelAxis = -1

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']

def densenet3d_model(input_shape=(dimz, dimx, dimy, channelNum),
                     num_outputs=4, 
                     n_base_filters=16,
                     growth_rate_k=16,                      
                     depth=5, 
                     dropout_rate=0.3,
                     optimizer=Adam, 
                     initial_learning_rate=5e-4,
                     kernel_reg_factor=1e-4,
                     ifbase = False,
                     ifcompile = False):
    
    # this model is channel last

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

        summation_layer, n_dense_filter = down_dense_block(in_conv, 3, n_level_filters, growth_rate_k, kernel=(3,3,3), 
                                                bottleneck=False, dropout_rate=dropout_rate, kernel_reg_factor=1e-4,
                                                activation=LeakyReLU, normMethod = 'instance_norm',
                                                grow_nb_filters=True, return_concat_list=False)        
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    output_layer = create_convolution_block(summation_layer, n_dense_filter, kernel=(1, 1, 1))
    pool1 = GlobalAveragePooling3D(data_format=data_format)(output_layer)

    if ifbase == True:
        model = Model(inputs=inputs, outputs=pool1)
        return model
    else: 
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                        kernel_initializer="he_normal",
                        activation="softmax",
                        kernel_regularizer=l2(kernel_reg_factor))(pool1)
        else:
            dense = Dense(units=num_outputs,
                        kernel_initializer="he_normal",
                        activation="sigmoid",
                        kernel_regularizer=l2(kernel_reg_factor))(pool1)
        model = Model(inputs=inputs, outputs=dense)
        if ifcompile == True:
            model.compile(optimizer=optimizer(lr=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), normMethod = 'instance_norm'):

    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if normMethod == 'batch_norm':
        layer = BatchNormalization(axis=channelAxis)(layer)
    elif normMethod == 'instance_norm':
        layer = InstanceNormalization(axis=channelAxis)(layer)
    elif normMethod == 'group_norm':
        layer = GroupNormalization(groups=4, axis=-1, epsilon=0.1)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def conv_block(layer, nb_filter, kernel=(3,3,3), bottleneck=False, dropout_rate=None, kernel_reg_factor=1e-4,
                activation=LeakyReLU, normMethod = 'instance_norm'):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        input_layer: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        kernel_reg_factor: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    if bottleneck:
        n_filters = nb_filter * 4  

        layer = Conv3D(n_filters, kernel=(1,1,1), kernel_initializer='he_normal', padding='same', use_bias=True,
                        kernel_regularizer=l2(kernel_reg_factor))(layer)
        if normMethod == 'batch_norm':
            layer = BatchNormalization(axis=channelAxis)(layer)
        elif normMethod == 'instance_norm':
            layer = InstanceNormalization(axis=channelAxis)(layer)
        elif normMethod == 'group_norm':
            layer = GroupNormalization(groups=4, axis=-1, epsilon=0.1)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)

    layer = Conv3D(nb_filter, kernel_size=kernel, kernel_initializer='he_normal', padding='same', use_bias=True)(layer)
    
    if normMethod == 'batch_norm':
        layer = BatchNormalization(axis=channelAxis)(layer)
    elif normMethod == 'instance_norm':
        layer = InstanceNormalization(axis=channelAxis)(layer)
    elif normMethod == 'group_norm':
        layer = GroupNormalization(groups=4, axis=-1, epsilon=0.1)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

    if dropout_rate:
        layer = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(layer)        
    return layer

def down_dense_block(layer, nb_layers, n_filters_k0, growth_rate_k, kernel=(3,3,3), bottleneck=False, dropout_rate=None, kernel_reg_factor=1e-4,
                  activation=LeakyReLU, normMethod = 'instance_norm', grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        layer: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        n_filters_k0: number of input filters
        growth_rate_k: growth rate
        kernel: kernel of the 3D conv
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        kernel_reg_factor: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    x_list = [layer]
    for i in range(nb_layers):
        convBlock = conv_block(layer, growth_rate_k, kernel, bottleneck,
                                dropout_rate, kernel_reg_factor, activation, 
                                normMethod)
        x_list.append(convBlock)

        layer = concatenate([layer, convBlock], axis=channelAxis)

        if grow_nb_filters:
            n_filters_k0 += growth_rate_k

    if return_concat_list:
        return layer, n_filters_k0, x_list
    else:
        return layer, n_filters_k0