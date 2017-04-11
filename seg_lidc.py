PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge,Activation
from keras.layers import AveragePooling3D, MaxPooling3D, Cropping3D,GlobalMaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import Adam,RMSprop
from keras.regularizers import l2
from deconv3D import Deconvolution3D
import preprocess1 as preprocess
import numpy as np
import sys
import os
import resnet3
import keras.backend as K
import tensorflow as tf

from matplotlib import pyplot as plt


BEST_WEIGHTS_PATH = 'models/best_seg_lidc.hdf5' 
REFINE_MODEL = 'models/refine_lidc.hdf5' 

DETECTIONS_PATH = preprocess.DETECTIONS_LIDC_ROOT+'detections.npy'

#cost = resnet3.dice_coef_loss
cost = binary_crossentropy
def conv_batch(prev,channels,kernel=3,stride=1,activation='relu',drop=0.0,name=None):
    conv = Convolution3D(channels,kernel,kernel,kernel,subsample=(stride,stride,stride),init='he_normal',
            W_regularizer=l2(0.00001),border_mode='same')(prev)
    conv = BatchNormalization()(conv)
    conv = Activation('relu',name=name)(conv)
    if drop>0:
        conv = Dropout(drop)(conv)
    return conv
def deconv_batch(prev,channels,kernel=3,stride=2,activation='relu'):
    shape = prev._keras_shape
    output_shape = (None,2*shape[1],2*shape[2],2*shape[3],channels)
    deconv = Deconvolution3D(channels,kernel,kernel,kernel,output_shape=output_shape,subsample=(stride,stride,stride),init='he_normal',border_mode='same',W_regularizer=l2(0.00001))(prev)

    deconv = BatchNormalization()(deconv)
    deconv = Activation('relu')(deconv)
    return deconv




def get_model(input_shape=preprocess.SEG_CROP+[1]):
    BASE = 16
 
    inputs = Input(shape=input_shape,dtype='float32',name='inputs') # n

    conv1a= conv_batch(inputs,BASE) 
    conv1b= conv_batch(conv1a,BASE)  
    pool1 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),border_mode='same')(conv1b) # n/2-2
 
    conv2a = conv_batch(pool1,2*BASE)
    conv2b = conv_batch(conv2a,2*BASE)  

    pool2 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),border_mode='same')(conv2b) #n/4-3
    conv3a = conv_batch(pool2,4*BASE)
    conv3b = conv_batch(conv3a,4*BASE) 
    pool3 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2),border_mode='same')(conv3b) #(n/4-7)/2 ->3
    conv4 = conv_batch(pool3,8*BASE,kernel=1,name='bottleneck') 
    deconv1 = deconv_batch(conv4,4*BASE) 
    res1 = resnet3.residual_block(deconv1,nb_filter=4*BASE)

    res1 = merge([res1,conv3b],mode='concat') 
    deconv2 = deconv_batch(res1,2*BASE)
    res2 = resnet3.residual_block(deconv2,nb_filter=2*BASE)
    res2 = merge([res2,conv2b],mode='concat') 
    deconv3 = deconv_batch(res2,BASE)
    res3 = resnet3.residual_block(deconv3,nb_filter=BASE)
    res3 = merge([res3,conv1b],mode='concat') 
    predictions1 = Convolution3D(1,1,1,1,activation='sigmoid',name='l1',init='normal')(res1)
    predictions2 = Convolution3D(1,1,1,1,activation='sigmoid',name='l2',init='normal')(res2)
    predictions3 = Convolution3D(1,1,1,1,activation='sigmoid',name='l3',init='normal')(res3)
    pool4 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv4)
    conv5 = resnet3.residual_block(pool4,nb_filter=4*BASE)
    pool5 = MaxPooling3D(pool_size=(3,3,3))(conv5)
    conv6 = conv_batch(pool5,150,kernel=1,name='pre-detect')
    detect = Convolution3D(1,1,1,1,activation='sigmoid',name='detect',init='normal')(conv6)
    model = Model(input=[inputs],output=[predictions1,predictions2,predictions3,detect])

    model.compile(optimizer=Adam(lr=1e-3),loss=cost,loss_weights=[0.2,0.3,0.5,0.2])
 
    print model.summary()
    return model

def main():
    REFINE = True
    print('Usage: %s <REFINE 1/0> PROCESSED_DIR/' % sys.argv[0])
    if len(sys.argv) == 3:
        REFINE = (sys.argv[1] == '1')
        preprocess.SEG_ROOT = sys.argv[2]
        if not os.path.exists('models'):
            os.mkdir('models')
    print('REFINE %s' % REFINE)

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
    N_VAL = 100
    if os.path.exists('quick-mode'):
        quick_mode = True
        N_VAL = len(data) // 2
    else:
        quick_mode = False

    data_t = data[:-N_VAL]
    labels_t = labels[:-N_VAL]
    data_v = data[-N_VAL:]
    labels_v = labels[-N_VAL:]
    if REFINE:
        print "Refining Model using previous detections"
        model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
        detections = np.load(DETECTIONS_PATH)
        detections_t = detections[:-N_VAL]
        detections_v = detections[-N_VAL:]
    else:
        detections = []
        detections_t = []
        detections_v = []

    if REFINE:

        save_best = ModelCheckpoint(REFINE_MODEL,save_best_only=True)
        model.compile(optimizer=Adam(lr=0.0002),loss = cost,loss_weights=[0.2,0.3,0.5,0.2])
        #model.compile(optimizer=Adam(lr=0.0001),loss = cost,loss_weights=[0.2,0.3,0.5])
        nb_epoch = 15
    else:
        save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)
        nb_epoch = 50

    if quick_mode:
        nb_epoch = 5

    BATCH_SIZE = 32
    train_generator= preprocess.generate_lidc_batch(data_t,labels_t,detections=detections_t,batch_size=BATCH_SIZE)
    valid_generator= preprocess.generate_lidc_batch(data_v,labels_v,detections=detections_v,batch_size=BATCH_SIZE)

    #exit()
    TRAIN = True
    #class_weight={0:1.,1:10.}
    #class_weight = {0:1.,1:1.}
    if TRAIN:
        model.fit_generator(generator= train_generator,validation_data=valid_generator,
                nb_val_samples=400,nb_epoch = nb_epoch, samples_per_epoch=4000,callbacks=[save_best],nb_worker=1) # nb_epoch = 40
        #if not REFINE:
        #    model.compile(optimizer=Adam(lr=0.0001),loss=cost,loss_weights=[0.2,0.3,0.5])
        #    train_generator= preprocess.generate_lidc_batch(data_t,labels_t,detections=detections_t,batch_size=BATCH_SIZE,neg_fraction=0.8)
        #    valid_generator= preprocess.generate_lidc_batch(data_v,labels_v,detections=detections_v,batch_size=1,neg_fraction=0.8)
        #    model.fit_generator(generator= train_generator,validation_data=valid_generator,
        #            nb_val_samples=400,nb_epoch=2,samples_per_epoch=4000,callbacks=[save_best],nb_worker=1)



 
    if REFINE:
        #model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
        model.load_weights(REFINE_MODEL,by_name=True)
    else:
        model.load_weights(BEST_WEIGHTS_PATH,by_name=True)

    if quick_mode:
        return

    test_generator= preprocess.generate_lidc_batch(data_v,labels_v,detections=detections_v,batch_size=1)

    data_test=[]
    mask_test=[]
    for _ in range(64):
        x,y=next(test_generator)
        data_test.append(x['inputs'])
        mask_test.append(y['l3'])

    data_test = np.concatenate(data_test,axis=0)
    mask_test = np.concatenate(mask_test,axis=0)
    predictions=model.predict(data_test,batch_size=BATCH_SIZE)
    predict=predictions[2]
    detect = predictions[3]
    print "Test",data_test.shape,predict.shape
    for i in range(len(data_test)):
        pix_sum = np.sum(mask_test[i,...])
        detected = np.max(detect[i,...])
        shape = data_test[i].shape
        print "nodule pixels:",pix_sum,detected,shape, np.max(predict[i,...])
        PLOT = False
        thresh = 0.15
        if PLOT and ((pix_sum>0.0 and detected<thresh) or (pix_sum == 0.0 and detected>thresh)) :
            big = 0
            ind1 = 0
            ind2 = 0
            ind3 = 0
            for k in range(mask_test.shape[1]):
                b = np.sum(mask_test[i,k,:,:,0])
                if b >= big:
                    ind1 = k
                    big = b
            big = 0
            for k in range(mask_test.shape[2]):
                b = np.sum(mask_test[i,:,k,:,0])
                if b >= big:
                    ind2 = k
                    big = b
            big = 0
            for k in range(mask_test.shape[3]):
                b = np.sum(mask_test[i,:,:,k,0])
                if b >= big:
                    ind3 = k
                    big = b
         
            plt.figure(1)
            plt.subplot(331)
            plt.imshow(np.squeeze(255*data_test[i,ind1,:,:,0]),cmap=plt.cm.gray)
            plt.subplot(332)
            plt.imshow(np.squeeze(255*mask_test[i,ind1,:,:,0]),cmap=plt.cm.gray)
            plt.subplot(333)
            plt.imshow(np.squeeze(255*predict[i,ind1,:,:,0]),cmap=plt.cm.gray)
            plt.subplot(334)
            plt.imshow(np.squeeze(255*data_test[i,:,ind2,:,0]),cmap=plt.cm.gray)
            plt.subplot(335)
            plt.imshow(np.squeeze(255*mask_test[i,:,ind2,:,0]),cmap=plt.cm.gray)
            plt.subplot(336)
            plt.imshow(np.squeeze(255*predict[i,:,ind2,:,0]),cmap=plt.cm.gray)
            plt.subplot(337)
            plt.imshow(np.squeeze(255*data_test[i,:,:,ind3,0]),cmap=plt.cm.gray)
            plt.subplot(338)
            plt.imshow(np.squeeze(255*mask_test[i,:,:,ind3,0]),cmap=plt.cm.gray)
            plt.subplot(339)
            plt.imshow(np.squeeze(255*predict[i,:,:,ind3,0]),cmap=plt.cm.gray)
         
            plt.show()

if __name__=='__main__':
    main()
