PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout,merge,MaxPooling3D,AveragePooling3D
import preprocess
import numpy as np
import resnet3
import keras.backend as K

def conv_batch(prev,channels,kernel=3,stride=2):
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',subsample=(stride,stride,stride),init='he_normal')(prev)
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',init='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',init='he_normal')(conv)
    conv = BatchNormalization()(conv)
    
    return conv


def get_model():
    resnet3.handle_dim_ordering()
    inputs = Input(shape=preprocess.NEW_SHAPE+(1,),dtype='float32')
    conv1 = resnet3._conv_bn_relu(nb_filter=64,nb_kernel=3,subsample=(2,2,2))(inputs)
    conv2 = resnet3._conv_bn_relu(nb_filter=128,nb_kernel=3,subsample=(2,2,2))(conv1)
    res1 = resnet3._residual_block(nb_filter=128,is_first_layer=True)(conv2)
    #res1 = resnet3._residual_block(resnet3.basic_block,nb_filter=128,is_first_layer=True)(res1)
 
    pool1 = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2))(res1)
    res2 = resnet3._residual_block(nb_filter=256,is_first_layer=True)(pool1)
    #res2 = resnet3._residual_block(resnet3.basic_block,nb_filter=256,is_first_layer=True)(res2)
 
    #last activation
    #res2 = resnet3._conv_bn_relu(nb_filter=512,nb_kernel=3)(res2)
    # Classifier block
    pool2 = AveragePooling3D(pool_size=(7,7,7),strides=(4,4,4))(res2)
    flatten = Flatten()(pool2)
    #flatten = Dropout(0.5)(flatten)
    predictions = Dense(1,activation='sigmoid',init='glorot_normal')(flatten)
    model = Model(input=inputs,output=predictions)
    model.compile(optimizer='adam',loss='binary_crossentropy')
 
    print model.summary()
    return model



model = get_model()   
print "loading data..."
data,labels = preprocess.load_numpy_images()
print "loading done"

model.fit(data,labels,batch_size=8,nb_epoch=50,verbose=1,validation_split=0.2)

