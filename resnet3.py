from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)

from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from  keras import initializations 

def my_he_init(shape,name=None,dim_ordering='tf'):
        return initializations.he_normal(shape,name=name,dim_ordering=dim_ordering)
def my_normal_init(shape,scale=0.05,name=None,dim_ordering='tf',mean = 0.):
        return K.random_normal_variable(shape,mean,scale,name=name)

def dice_coef(y_true, y_pred):
    SMOOTH = 1.
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f,axis=-1)
    union = K.sum(y_true_f,axis=-1) + K.sum(y_pred_f,axis=-1)
    return K.mean((2. * intersection + SMOOTH) / (union + SMOOTH),axis=-1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def shortcut(input, residual,init_subsample):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """

    equal_channels = residual._keras_shape[-1] == input._keras_shape[-1]
    # 1 X 1 conv if shape is different. Else identity.
    if init_subsample != (1,1,1)  or not equal_channels:
        shortcut = Convolution3D(residual._keras_shape[-1],
                                 1,1,1,
                                 subsample=init_subsample,
                                 init='he_normal', border_mode="same")(input)
    else:
        shortcut = input
    shortcut = BatchNormalization()(shortcut)
    return merge([shortcut, residual], mode="sum")

def residual_block(input,nb_filter, init_subsample=(1,1,1),border_mode='same'):
    conv1 = Convolution3D(nb_filter,3,3,3, subsample=init_subsample,
                         init='he_normal', border_mode=border_mode)(input)

    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Convolution3D(nb_filter,3,3,3,init='he_normal', border_mode=border_mode)(conv1)
    conv2 = BatchNormalization()(conv2)
    output =  shortcut(input, conv2,init_subsample)

    output= Activation('relu')(output)

    return output



