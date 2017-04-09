PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge, Activation, Lambda
from keras.layers import AveragePooling3D, MaxPooling3D, TimeDistributed, GlobalAveragePooling3D,GlobalMaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD,Adam
from keras import initializations
from keras.regularizers import l2
import preprocess1 as pre
import os
import sys
import numpy as np
import resnet3
import keras.backend as K
import tensorflow as tf
#import seg1
import densenet3
import pandas as pd
import random

def my_init(shape, name=None,dim_ordering='tf'):
        return initializations.normal(shape, scale=0.001, name=name)

def conv_batch(prev,channels,kernel=3,stride=1,activation='relu',drop=0.0,name=None,init='he_normal'):
    conv = Convolution3D(channels,kernel,kernel,kernel,subsample=(stride,stride,stride),init=init,
            W_regularizer=l2(1e-5),border_mode='valid')(prev)
    conv = BatchNormalization()(conv)
    conv = Activation('relu',name=name)(conv)
    if drop>0:
        conv = Dropout(drop)(conv)
    return conv


def get_model():
#    model = densenet3.create_dense_net(nb_classes=2, img_dim=tuple(pre.SEG_CROP)+(5,), nb_dense_block=4,nb_layers=4, growth_rate=12, bottleneck=True, reduction=0.5,dropout_rate=0.1)
    model = densenet3.create_dense_net(nb_classes=1, img_dim= pre.det_crop+(160,), nb_dense_block=2,nb_layers=2, growth_rate=16, bottleneck=True, reduction=0.5,dropout_rate=0.2)

#    model_input = Input(shape=(5,5,5,160),name='inputs')
#    n = BatchNormalization()(model_input)
#    pool = GlobalAveragePooling3D()(n)
#    drop = Dropout(0.5)(pool)
#    out = Dense(1,activation='sigmoid',name='output')(drop)
#    model = Model(input = model_input,output = out)
    model.summary()
    return model
def get_model1():

    inputs = Input(shape=pre.det_crop+(160,),dtype='float32',name='inputs')
    b = BatchNormalization()(inputs)	
    pool4 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(b)
    conv5 = resnet3.residual_block(pool4,nb_filter=128)
    #conv5 = seg1.narrow(pool4,128)
    pool5 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv5)
    conv6 = resnet3.residual_block(pool5,nb_filter=64)
    #conv6 = seg1.narrow(pool5,64)
    pool6 = MaxPooling3D(pool_size=(4,4,4),strides=(2,2,2))(conv6)
    flat = Flatten()(pool6)
    flat = Dropout(0.5)(flat)
    predictions = Dense(1,activation='sigmoid',init='normal',name='output')(flat)
    model = Model(input=inputs,output=predictions)
    model.summary()
    return model

def get_model2():

    inputs = Input(shape=pre.det_crop+(160,),dtype='float32',name='inputs')
    b = BatchNormalization()(inputs)	
    pool4 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(b)
    conv5 = resnet3.residual_block(pool4,nb_filter=128)
    #conv6 = resnet3.residual_block(conv5,nb_filter=128)

    #conv5 = seg1.narrow(pool4,128)
    pool5 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv5)
    conv6 = resnet3.residual_block(pool5,nb_filter=64)
    #conv6 = seg1.narrow(pool5,64)
    #pool6 = MaxPooling3D(pool_size=(4,4,4),strides=(2,2,2))(conv6)
    #flat = Flatten()(pool6)
    flat = GlobalMaxPooling3D()(conv6)
    flat = Dropout(0.5)(flat)
    def my_init(shape, name=None):
            return initializations.normal(shape, scale=0.01, name=name)
    predictions = Dense(1,activation='sigmoid',init=my_init,name='output')(flat)
    model = Model(input=inputs,output=predictions)
    model.summary()
    return model

def get_model3():

    inputs = Input(shape=pre.det_crop+(151,),dtype='float32',name='inputs')
    conv5 = resnet3.residual_block(inputs,nb_filter=300) #64
    conv5= Dropout(0.2)(conv5)
    flat = GlobalMaxPooling3D()(conv5)
    flat = Dropout(0.5)(flat)

    predictions = Dense(1,activation='sigmoid',init='normal',name='output')(flat)
    model = Model(input=inputs,output=predictions)
    model.summary()
    return model

def get_model4():

    inputs = Input(shape=pre.det_crop+(151,),dtype='float32',name='inputs')
    conv5 = resnet3.residual_block(inputs,nb_filter=150) #64
    conv5= Dropout(0.3)(conv5)
    flat = GlobalMaxPooling3D()(conv5)
    flat = Dropout(0.5)(flat)

    predictions = Dense(1,activation='sigmoid',init='normal',name='output')(flat)
    model = Model(input=inputs,output=predictions)
    model.summary()
    return model

def log_loss(t, y):
    eps = 1e-15
    y = np.clip(y, eps, 1-eps)
    return -np.mean(t*np.log(y) + (1 - t)*np.log(1 - y))

print('Usage: %s ORIGINAL_IMAGES_ROOT/ PROCESSED_IMAGES_ROOT/ LABELS_FILE' % sys.argv[0])
if len(sys.argv) == 4:
    pre.ORIGINAL_IMAGES_ROOT = sys.argv[1]
    pre.PROCESSED_IMAGES_ROOT = sys.argv[2]
    pre.LABELS_FILE = sys.argv[3]

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
K.set_session(session)

model = get_model4()
model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])


if not os.path.exists('logs'):
    os.mkdir('logs')
BEST_WEIGHTS_PATH = 'logs/detect_best_weights.hdf5'
TRAIN = True
if TRAIN:
    print "loading train data..."
   
    data,labels = pre.load_numpy_detections()
    split=len(labels)/10
    #split = 1
    print 'validation size',split
    data_train = data[:-split]
    labels_train = labels[:-split]
    data_v = data[-split:]
    labels_v = labels[-split:]
    print "loading done"


    save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)


    train_gen = pre.generate_detect_batch(data_train,labels_train,rand=True,batch_size=52) #34,26

    val_gen = pre.generate_detect_batch(data_v,labels_v,rand=False,batch_size=52)


    print "training labels",len(labels_train)

    model.fit_generator(generator=train_gen,samples_per_epoch=len(labels_train),nb_epoch=50, validation_data=val_gen,nb_val_samples=split,callbacks=[save_best],nb_worker=1,verbose=1)


# SUBMIT
model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
if True:
    val_predictions = model.predict(data_v, batch_size=1)

    val_loss = log_loss(labels_v, np.squeeze(val_predictions))
    print('val mean %.4f loss %.4f' % (val_predictions.mean(), val_loss))


data_test,ids = pre.load_numpy_detections(dataset='test')
print "predicting..."

predictions = model.predict(data_test,batch_size=1,verbose=1)

df = pd.DataFrame({'id':pd.Series(ids),'cancer':pd.Series(np.squeeze(predictions))})
df.to_csv('predictions.csv',header=True,columns=['id','cancer'],index=False)



