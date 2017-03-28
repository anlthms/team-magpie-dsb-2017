from keras.layers import (
    Activation, Convolution3D, Dropout, MaxPooling3D, merge, BatchNormalization,AveragePooling3D,
    Input)
from deconv3D import Deconvolution3D
from keras.regularizers import l2
from keras import backend as K

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2,weight_decay=1e-5):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    """

    l = BatchNormalization(gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(inputs)
    l = Activation('relu')(l)
    l = Convolution3D(n_filters, filter_size, filter_size,filter_size,border_mode='same', init='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def CropPooling3D(inputs,crop=10):
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l1 = AveragePooling3D((2,2,2), strides=(2,2,2))(l)
    l2 = Cropping3d(cropping=((crop,crop,crop),(crop,crop,crop),(crop,crop,crop)))(l)
    l = merge([l1,l2],mode='concat')
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2,to_crop=False):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = AveragePooling3D((2,2,2), strides=(2,2,2))(l)
    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = merge(block_to_upsample,mode='concat')
    
    shape = l._keras_shape
    #print 'shape',shape
    output_shape=(None,2*shape[1],2*shape[2],2*shape[3],n_filters_keep)
    l = Deconvolution3D(n_filters_keep, 3,3,3, subsample=(2,2,2),output_shape=output_shape, init='he_uniform',
            border_mode='same')(l)
    # Concatenate with skip connection
    l = merge([l, skip_connection],mode='concat') # cropping=[None, None, 'center', 'center']

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution


def SigmoidLayer(inputs,name='output'):
    """
    Performs 1x1 convolution followed by sigmoid nonlinearity
    """

    l = Convolution3D (1,1,1,1, init='he_uniform', border_mode='same',activation='sigmoid',name=name)(inputs)


    return l

    # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
    # any improvements. Our guess is that FC-DenseNet naturally permits this multiscale approach
