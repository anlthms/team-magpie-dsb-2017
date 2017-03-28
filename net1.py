PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge
import preprocess
import numpy as np

def conv_batch(prev,channels,kernel=3,stride=2):
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',subsample=(stride,stride,stride),init='he_normal')(prev)
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',init='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Convolution3D(channels,kernel,kernel,kernel,activation='relu',init='he_normal')(conv)
    conv = BatchNormalization()(conv)
    
    return conv



def get_model():

    inputs = Input(shape=preprocess.NEW_SHAPE+(1,),dtype='float32')
    conv1 = conv_batch(inputs,64)
    conv2 = conv_batch(conv1,128)
    conv3 = conv_batch(conv2,256,stride=4)
 
    flat = Flatten()(conv3)
    #flat = merge([flat1,flat2],mode='concat',concat_axis=1)
    dense1 = Dense(512,activation='relu',init='glorot_normal')(flat)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(512,activation='relu',init='glorot_normal')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.5)(dense2)
    predictions = Dense(1,activation='sigmoid',init='glorot_normal')(dense2)
    model = Model(input=inputs,output=predictions)
    model.compile(optimizer='adam',loss='binary_crossentropy')
    print model.summary()
    return model
    
model = get_model()   
print "loading data..."
data,labels = preprocess.load_numpy_images()
print "loading done"
model.fit(data,labels,batch_size=8,nb_epoch=50,verbose=1,validation_split=0.1)

