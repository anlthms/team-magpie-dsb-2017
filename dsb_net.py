PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge
from keras.layers import AveragePooling3D, MaxPooling3D,Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD,Adam
from keras.initializations import normal
from sklearn.metrics import log_loss
import preprocess1 as preprocess
import numpy as np
import keras.backend as K
import tensorflow as tf
import lidc
import resnet3
import pandas as pd
import random
from tqdm import tqdm


def conv_batch(prev,channels,kernel=3,stride=1,activation='relu',border_mode='valid'):
    conv = Convolution3D(channels,kernel,kernel,kernel,subsample=(stride,stride,stride),init='he_normal',border_mode = border_mode)(prev)

    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv


def predict_test(data):
    crop_data=[]
    for i in range(len(data)):
        icrop = preprocess.crop_center(np.squeeze(data[i]))
        crop_data.append(icrop)
    crop_data = np.asarray(crop_data)
    crop_data = np.expand_dims(crop_data,axis=4)
    predictions = model.predict(crop_data,batch_size=1,verbose=1)
    return predictions

def augment(image):
    images = []
    x = (image.shape[0]-preprocess1.NEW_SHAPE_CROP[0])/2
    y = (image.shape[1]-preprocess1.NEW_SHAPE_CROP[1])/2
    z = (image.shape[2]-preprocess1.NEW_SHAPE_CROP[2])/2
    for dx in [-3,3]:
	for dy in [-3,3]:	
            for dz in [-3,3]:
                image_crop = preprocess.seg_crop(image,(x+dx,y+dy,z+dz),preprocess.NEW_SHAPE_CROP)
                images.append(image_crop)
    return np.asarray(images)

def predict_augment(model,images):
    predictions=[]
    for image in tqdm(images):
        augmented = augment(image)
        predictions1 = model.predict(augmented,batch_size=4)
        prediction = np.exp(np.mean(np.log(predictions1)))
        predictions.append(prediction)
    predictions = np.asarray(predictions)
   
    return predictions

def get_model():
    resnet3.handle_dim_ordering()
    model = lidc.get_model(preprocess.NEW_SHAPE+[1])
    model.load_weights(lidc.BEST_WEIGHTS_PATH,by_name=True)
    #for layer in model.layers:
        #layer.trainable=False
    #nodule1 = model.get_layer('nodule1').output
    #malig1 = model.get_layer('malig1').output
    #diameter1 = model.get_layer('diameter1').output
    #pre = model.get_layer('pre_prediction').output

    #z = merge([pre,nodule1,malig1,diameter1],mode='concat')
    #conv4 = resnet3.residual_block(z,nb_filter=128)
    z = model.get_layer('activation_7').output
    conv4 = conv_batch(z,128)
    pool4 = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2),border_mode='valid')(conv4)
    #conv5 = resnet3.residual_block(pool4,nb_filter=128)
    conv5 = conv_batch(pool4,128) 	
    pool5 = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2))(conv5)
    #conv6 = conv_batch(pool6,128) 	
    #pool6 = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2))(conv6)
 
    flat = Flatten()(pool5)
    #flat = Dropout(0.5)(flat)
    def my_init(shape,name=None):
        return normal(shape, scale=0.001, name=name)
    adam = Adam(lr=0.001) 
      
    predictions = Dense(1,activation='sigmoid',name='output',init=my_init)(flat)
    model = Model(input=model.input,output=predictions)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    print model.summary()
    return model
    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
K.set_session(session)
model = get_model()  


random.seed(1)
print "loading train data..."
data,labels = preprocess.load_numpy_images()
p = range(len(labels))
random.shuffle(p) # inplace
data = data[p]
labels = labels[p]
print data.shape,labels.shape
split=len(labels)/5
print 'validation size',split
data_train = data[:-split]
labels_train = labels[:-split]
data_v = data[-split:]
labels_v = labels[-split:]
print data_v.shape,labels_v.shape
print "loading done"

BEST_WEIGHTS_PATH = 'logs/final_best_weights.hdf5'

save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)


TRAIN = True
if TRAIN:

    model.fit_generator(generator=preprocess.generate_dsb_batch(data_train,labels_train,rand=True,batch_size=2),
        samples_per_epoch=1260,nb_epoch=20, 
        validation_data=preprocess.generate_dsb_batch(data_v,labels_v,rand=False,batch_size=2),
        verbose=1,nb_val_samples=split,callbacks=[save_best],nb_worker=1)

    model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
"""
    for layer in model.layers:
            layer.trainable=True
    adam = Adam(lr=0.0001) 
    model.compile(optimizer=adam ,loss='binary_crossentropy',metrics=['accuracy'])
     
    model.fit_generator(generator=preprocess.generate_dsb_batch(data_train,labels_train,rand=True,batch_size=3),
        samples_per_epoch=1260,nb_epoch=50,
        validation_data=preprocess.generate_dsb_batch(data_v,labels_v,rand=False,batch_size=1),verbose=1,nb_val_samples=split,callbacks=[checkpoint,save_best],nb_worker=1)
"""
exit()
# SUBMIT
model.load_weights(BEST_WEIGHTS_PATH,by_name=True)

CHECK_VALID = True

if CHECK_VALID:
    predictions = predict_test(data_v)

    metric = log_loss(labels_v,predictions)
    print "validation result:",metric

    #aug_predictions = predict_augment(model,data_v)
    #metric = log_loss(labels_v,aug_predictions)
    #print "augmented validation result:", metric

print "loading submission data..."
 
data_test,ids = preprocess.load_numpy_images(dataset='test')
print "predicting..."
#predictions = model.predict(data_test,batch_size=1,verbose=1)



#predictions = predict_augment(model,data_test)
predictions = predict_test(data_test)
df = pd.DataFrame({'id':pd.Series(ids),'cancer':pd.Series(np.squeeze(predictions))})
df.to_csv('predictions.csv',header=True,columns=['id','cancer'],index=False)
    

    

     


