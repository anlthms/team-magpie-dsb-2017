PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD,Adam
from keras.initializations import normal
import preprocess
import numpy as np
import resnet3
import keras.backend as K
import tensorflow as tf
import seg
import resnet3
import pandas as pd

def get_model():
    resnet3.handle_dim_ordering()
    #inputs = Input(shape=preprocess.NEW_SHAPE+(1,),dtype='float32',name='inputs')
    seg_model = seg.get_model(preprocess.NEW_SHAPE_CROP+(1,))
    seg_model.load_weights(seg.BEST_WEIGHTS_PATH,by_name=True)
    for layer in seg_model.layers:
        layer.trainable=False
    print "Get fine-tuned model"
    x=seg_model.get_layer('merge_3').output # with older resnet (activation before merge)
    
    pool4 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(x)
    conv5 = resnet3.residual_block(pool4,nb_filter=128)
    pool5 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv5)
    conv6 = resnet3.residual_block(pool5,nb_filter=64)
    pool6 = MaxPooling3D(pool_size=(4,4,4))(conv6)
    flat = Flatten()(pool6)
    flat = Dropout(0.5)(flat)
    def my_init(shape,name=None):
        return normal(shape, scale=0.001, name=name)
    adam = Adam(lr=0.0005) 
      
    predictions = Dense(1,activation='sigmoid',init=my_init,name='output')(flat)
    model = Model(input=seg_model.input,output=predictions)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    print model.summary()
    return model
    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
K.set_session(session)
model = get_model()  

TRAIN = True
if TRAIN:
    print "loading train data..."
    data,labels = preprocess.load_numpy_images()
    print data.shape
    split=len(labels)/10
    print 'validation size',split
    data_train = data[:-split]
    labels_train = labels[:-split]
    data_v = data[-split:]
    labels_v = labels[-split:]
    print data_v.shape,labels_v.shape
    print "loading done"

BEST_WEIGHTS_PATH = 'logs/final_best_weights.hdf5'

checkpoint = ModelCheckpoint('logs/final_weights.{epoch:02d}--{val_loss:.2f}.hdf5')
save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)

#model.fit(data,labels,batch_size=4,validation_split=0.1,nb_epoch=50,callbacks=[checkpoint,save_best])


if TRAIN:

    model.fit_generator(generator=preprocess.generate_dsb_batch(data_train,labels_train,rand=True,batch_size=4),
        samples_per_epoch=1260,nb_epoch=10, validation_data=preprocess.generate_dsb_batch(data_v,labels_v,rand=False,batch_size=1),verbose=1,nb_val_samples=split,callbacks=[checkpoint,save_best],nb_worker=1)

    model.load_weights(BEST_WEIGHTS_PATH,by_name=True)

    for layer in model.layers:
            layer.trainable=True
    adam = Adam(lr=0.0001) 
    model.compile(optimizer=adam ,loss='binary_crossentropy',metrics=['accuracy'])
     
    model.fit_generator(generator=preprocess.generate_dsb_batch(data_train,labels_train,rand=True,batch_size=3),
        samples_per_epoch=1260,nb_epoch=20,
        validation_data=preprocess.generate_dsb_batch(data_v,labels_v,rand=False,batch_size=1),verbose=1,nb_val_samples=split,callbacks=[checkpoint,save_best],nb_worker=1)


# SUBMIT
model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
print "loading submission data..."
 
data_test,ids = preprocess.load_numpy_images(dataset='test')
print "predicting..."
predictions = model.predict(data_test,batch_size=1,verbose=1)
df = pd.DataFrame({'id':pd.Series(ids),'cancer':pd.Series(np.squeeze(predictions))})
df.to_csv('predictions.csv',header=True,columns=['id','cancer'],index=False)
    

    

     


