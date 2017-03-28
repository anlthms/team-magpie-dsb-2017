PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge
from keras.layers import AveragePooling3D, MaxPooling3D
import preprocess
import numpy as np
import resnet3
import keras.backend as K
import tensorflow as tf


def conv_batch(prev,channels,kernel,stride=1):
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',subsample=(stride,stride,stride),init='he_normal')(prev)
    conv = BatchNormalization()(conv)
    
    return conv



def get_model():
    resnet3.handle_dim_ordering()
    inputs = Input(shape=preprocess.NEW_SHAPE+(1,),dtype='float32')
    conv1 = conv_batch(inputs,16,kernel=3)
    conv2 = conv_batch(conv1,16,kernel=3)
 
    pool2 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv2)

    conv2 = conv_batch(pool2,32,kernel=3)
    conv3 = conv_batch(conv2,32,kernel=3)
    conv4 = conv_batch(conv3,32,kernel=3)
    # max/avg pooling here ?
    conv5 = conv_batch(conv4,64,kernel=1)
    
    conv6 = conv_batch(conv5,4,kernel=1)
    pool6 = AveragePooling3D(pool_size=(11,11,11),strides=(10,10,10))(conv6)
    flat = Flatten()(pool6)

    #flat = merge([flat1,flat2],mode='concat',concat_axis=1)
    dense1 = Dense(128,activation='relu',init='glorot_normal')(flat)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    predictions = Dense(1,activation='sigmoid',init='glorot_normal')(dense1)
    model = Model(input=inputs,output=predictions)
    model.compile(optimizer='adam',loss='binary_crossentropy')
    print model.summary()
    return model
    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
K.set_session(session)
model = get_model()   
print "loading data..."
data,labels = preprocess.load_numpy_images()
print "loading done"
model.fit(data,labels,batch_size=4,nb_epoch=50,verbose=1,validation_split=0.1)

