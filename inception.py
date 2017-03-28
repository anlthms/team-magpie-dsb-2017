from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import MaxPooling3D, Convolution3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

def block_A(input,scale_residual=True):
    
    # Input is relu activation
    init = input

    ir1 = Convolution3D(24, 1, 1,1, border_mode='same',init='he_normal')(input)
    ir1 = BatchNormalization()(ir1)
    ir1 = Activation('relu')(ir1)

    ir2 = Convolution3D(24, 1, 1, 1, border_mode='same',init='he_normal')(input)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
	
    ir2 = Convolution3D(24, 3, 3, 3,border_mode='same',init='he_normal')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
	 	

    ir3 = Convolution3D(24, 1, 1, 1, border_mode='same',init='he_normal')(input)
    ir3 = BatchNormalization()(ir3)
    ir3 = Activation('relu')(ir3)
	
    ir3 = Convolution3D(24, 3, 3, 3, border_mode='same',init='he_normal')(ir3)
    ir3 = BatchNormalization()(ir3)
    ir3 = Activation('relu')(ir3)
	
    ir3 = Convolution3D(24, 3, 3, 3, border_mode='same',init='he_normal')(ir3)
    ir3 = BatchNormalization()(ir3)
    ir3 = Activation('relu')(ir3)
	
    ir_merge = merge([ir1, ir2, ir3], concat_axis=-1, mode='concat')

    ir_conv = Convolution3D(input._keras_shape[-1], 1,1, 1, border_mode='same',init='he_normal')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def block_B(input, scale_residual=True):

    init = input

    ir1 = Convolution3D(48, 1, 1, 1,border_mode='same',init='he_normal')(input)
    ir1 = BatchNormalization()(ir1)
    ir1 = Activation('relu')(ir1)
    ir2 = Convolution3D(48, 1, 1, 1,border_mode='same',init='he_normal')(input)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
    ir2 = Convolution3D(48, 1, 1, 7,border_mode='same',init='he_normal')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
    ir2 = Convolution3D(48, 1, 7, 1,border_mode='same',init='he_normal')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
    ir2 = Convolution3D(48, 7, 1, 1,border_mode='same',init='he_normal')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat')

    ir_conv = Convolution3D(97, 1, 1, 1, border_mode='same',init='he_normal')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def block_C(input, scale_residual=True):
    init = input

    ir1 = Convolution3D(96, 1, 1,1, border_mode='same')(input)
    ir1 = BatchNormalization()(ir1)
    ir1 = Activation('relu')(ir1)
    ir1 = Convolution3D(96, 1, 1,1, border_mode='same')(input)
    ir1 = BatchNormalization()(ir1)
    ir1 = Activation('relu')(ir1)
 

    ir2 = Convolution3D(96, 1, 1,1, border_mode='same')(input)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
 
    ir2 = Convolution3D(96, 1, 1,3, border_mode='same')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
 
    ir2 = Convolution3D(96, 1, 3,1, border_mode='same')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
 
    ir2 = Convolution3D(96, 3, 1, 1,border_mode='same')(ir2)
    ir2 = BatchNormalization()(ir2)
    ir2 = Activation('relu')(ir2)
 


    ir_merge = merge([ir1, ir2], mode='concat')

    ir_conv = Convolution3D(385, 1, 1, 1,border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out



def reduction_A(input, k=48, l=48, m=48, n=48):

    r1 = MaxPooling3D((3,3,3), strides=(2,2,2),border_mode='same')(input)

    r2 = Convolution3D(n, 3, 3 ,3, subsample=(2,2,2),init='he_normal',border_mode='same')(input)
    r2 = BatchNormalization()(r2)
    r2 = Activation('relu')(r2)
    r3 = Convolution3D(k, 1, 1, 1, border_mode='same',init='he_normal')(input)
    r3 = BatchNormalization()(r3)
    r3 = Activation('relu')(r3)
    r3 = Convolution3D(l, 3, 3, 3, border_mode='same',init='he_normal')(r3)
    r3 = BatchNormalization()(r3)
    r3 = Activation('relu')(r3)
    r3 = Convolution3D(m, 3, 3, 3, border_mode='same',subsample=(2,2,2),init='he_normal')(r3)
    r3 = BatchNormalization()(r3)
    r3 = Activation('relu')(r3)
  
    a = merge([r1, r2, r3], mode='concat')
    return a

def reduction_B(input):

    r1 = MaxPooling3D((3,3,3), strides=(2,2,2), border_mode='same')(input)

    r2 = Convolution3D(96, 1, 1,1, border_mode='same')(input)
    r2 = BatchNormalization()(r2)
    r2 = Activation('relu')(r2) 
    r2 = Convolution3D(96, 3, 3,3, subsample=(2,2,2),border_mode='same')(r2)
    r2 = BatchNormalization()(r2)
    r2 = Activation('relu')(r2) 
 
    r3 = Convolution3D(96, 1, 1,1, border_mode='same')(input)
    r3 = BatchNormalization()(r3)
    r3 = Activation('relu')(r3) 
    r3 = Convolution3D(96, 3, 3, 3, subsample=(2, 2,2),border_mode='same')(r3)
    r3 = BatchNormalization()(r3)
    r3 = Activation('relu')(r3) 
 
    r4 = Convolution3D(96, 1, 1,1, border_mode='same')(input)
    r4 = BatchNormalization()(r4)
    r4 = Activation('relu')(r4) 
 
    r4 = Convolution3D(96, 3, 3,3, border_mode='same')(r4)
    r4 = BatchNormalization()(r4)
    r4 = Activation('relu')(r4) 
 
    r4 = Convolution3D(96, 3, 3,3, subsample=(2, 2,2),border_mode='same')(r4)
    r4 = BatchNormalization()(r4)
    r4 = Activation('relu')(r4) 
 
    m = merge([r1, r2, r3, r4], mode='concat')
    return m

