PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import (Input, Dense, Convolution3D,Flatten, BatchNormalization, 
        Dropout, merge, Activation)
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.objectives import binary_crossentropy,mse
from keras.metrics import binary_accuracy
from deconv3D import Deconvolution3D
import preprocess_lidc as preprocess
import numpy as np
import resnet3
import keras.backend as K
import tensorflow as tf

from matplotlib import pyplot as plt


BEST_WEIGHTS_PATH = 'models/best_lidc.hdf5' 

def custom_loss(y_true,y_pred):
    # nodule,malig,diameter
    l1 = custom_nodule_loss(y_true,y_pred)
    l2 = custom_malig_loss(y_true_y,y_pred)
    l3 = custom_diameter_loss(y_true,y_pred)
    return l1+l2+l3

def custom_nodule_loss(y_true,y_pred):
    return mse(y_true[:,0],y_pred[:,0])
def custom_malig_loss(y_true,y_pred):
    return custom_mse_loss(y_true,y_pred,1)
def custom_diameter_loss(y_true,y_pred):
    return custom_mse_loss(y_true,y_pred,2)

def custom_mse_loss(y_true,y_pred,k):
    z = preprocess.NODULE_THRESHOLD*K.ones_like(y_true[:,0])
    include = K.cast(K.greater(y_true[:,0],z),'float32')
    diff = y_true[:,k]-y_pred[:,k]
    l2 = K.square(diff[:,0])
    l2 = K.sum(include*l2,axis=-1)/(K.sum(include)+1e-6)
    return l2



def conv_batch(prev,channels,kernel=3,stride=1,activation='relu',border_mode='valid'):
    conv = Convolution3D(channels,kernel,kernel,kernel,subsample=(stride,stride,stride),init='he_normal',border_mode = border_mode)(prev)

    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def get_model(input_shape = preprocess.CROP_SHAPE+(1,)):
    resnet3.handle_dim_ordering()
    inputs = Input(shape=input_shape,dtype='float32',name='inputs')
    conv1 = conv_batch(inputs,32,stride=2)
    #pool1 = MaxPooling3D((3,3,3),strides=(2,2,2))(inputs)
    #conv1b = conv_batch(pool1,16,kernel=1)
    #conv1 = merge((conv1a,conv1b),mode='concat')
    #conv1 = conv_batch(conv1,32,border_mode='same')
    conv2 = resnet3.residual_block(conv1,64)
    #conv2 = conv_batch(conv1,32,border_mode='same')
    conv2 = MaxPooling3D((3,3,3),strides=(2,2,2))(conv2)
    conv3 = resnet3.residual_block(conv2,128)
    conv3 = MaxPooling3D((3,3,3),strides=(2,2,2))(conv3)
    conv4 = resnet3.residual_block(conv3,128)
    #conv4 = Flatten()(conv4)
    pool4 = MaxPooling3D((3,3,3),strides=(3,3,3))(conv4)
    #pool4 = Flatten()(pool4)
    pool4 = Dropout(0.5)(pool4)
    # first output is nodule presnece, second is malig_score/5
    #dense1 = Dense(256,activation='relu')(pool4)
    dense1 = Convolution3D(150,1,1,1,activation='relu',name='pre_prediction')(pool4)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    prediction_nodule1 = Convolution3D(1,1,1,1,activation='sigmoid',name='nodule1')(dense1)
    prediction_malig1 = Convolution3D(1,1,1,1,activation='sigmoid',name='malig1')(dense1)
    prediction_diameter1 = Convolution3D(1,1,1,1,activation='relu',name='diameter1')(dense1)
    prediction_nodule = Flatten(name='nodule')(prediction_nodule1)
    prediction_malig = Flatten(name='malig')(prediction_malig1)
    prediction_diameter = Flatten(name='diameter')(prediction_diameter1)
    model = Model(input=[inputs],output=[prediction_nodule,prediction_malig,prediction_diameter])
    model.compile(optimizer='adam',loss = [mse,mse,mse],loss_weights=[2.,2.,1.])
 
    print model.summary()
    return model

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    K.set_session(session)

    RESUME = False
    if RESUME:
        model = load_model(BEST_MODEL_PATH)
    else:
        model = get_model()   

    print "loading data..."
    data,labels = preprocess.load_lidc()
    N_VAL = 50
    data_t = data[:-N_VAL]
    labels_t = labels[:-N_VAL]
    data_v = data[-N_VAL:]
    labels_v = labels[-N_VAL:]

    save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)


    BATCH_SIZE = 32
    train_generator= preprocess.generate_lidc_batch(data_t,labels_t,batch_size=BATCH_SIZE)
    valid_generator= preprocess.generate_lidc_batch(data_v,labels_v,batch_size=BATCH_SIZE)

    #exit()
    TRAIN = True
    if TRAIN:
        model.fit_generator(generator= train_generator,validation_data=valid_generator,
                nb_val_samples=1200,nb_epoch=50,samples_per_epoch=6000,callbacks=[save_best],nb_worker=1)


    model.load_weights(BEST_WEIGHTS_PATH,by_name=True)

    test_generator = preprocess.generate_lidc_batch(data_v,labels_v,batch_size=16)
    data_check,labels_check = test_generator.next()
    predictions = model.predict(data_check['inputs'],batch_size=16)
       
    for i,image in enumerate(data_check['inputs']):
        true_nodule = labels_check['nodule'][i]
        true_malig = labels_check['malig'][i]
        true_diameter = labels_check['diameter'][i]
        #print i
        predicted_nodule = predictions[0][i]
        predicted_malig = predictions[1][i]
        predicted_diameter = predictions[2][i]
        #print [true_nodule,true_malig,true_diameter],[predicted_nodule,predicted_malig,predicted_diameter]


if __name__=='__main__':
    main()
