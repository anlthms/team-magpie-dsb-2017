PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge,Activation
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from deconv3D import Deconvolution3D
import preprocess
import numpy as np
import resnet3
import keras.backend as K
import tensorflow as tf
import inception

from matplotlib import pyplot as plt


BEST_WEIGHTS_PATH = 'logs/best_weights_seg2.hdf5' 


def conv_batch(prev,channels,kernel=3,stride=1,activation='relu'):
    conv = Convolution3D(channels,kernel,kernel,kernel,subsample=(stride,stride,stride),init='he_normal',border_mode='same')(prev)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)   
    return conv
def deconv_batch(prev,channels,kernel=3,stride=2,activation='relu'):
    shape = prev._keras_shape
    #if shape[2] is None:
    #    output_shape=(None,None,None,None,channels)
    #else:
    output_shape = (None,2*shape[1],2*shape[2],2*shape[3],channels)
    conv = Deconvolution3D(channels,kernel,kernel,kernel,output_shape=output_shape,subsample=(stride,stride,stride),init='he_normal',border_mode='same')(prev)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def get_model(input_shape=(preprocess.SEG_CROP,preprocess.SEG_CROP,preprocess.SEG_CROP,1)):
    BASE = 32
    resnet3.handle_dim_ordering()
    #inputs = Input(shape=preprocess.NEW_SHAPE+(1,),dtype='float32',name='inputs')
    #inputs = Input(shape=(None,None,None,1),dtype='float32',name='inputs')
 
    inputs = Input(shape=input_shape,dtype='float32',name='inputs')
    conv1 = inception.block_A(inputs)
    reduce1 = inception.reduction_A(conv1)
    conv2 =  inception.block_B(reduce1)
    reduce2 = inception.reduction_B(conv2)
    conv3 = inception.block_C(reduce2)
    pool3 = MaxPooling3D((3,3,3),strides=(2,2,2),border_mode='same')(conv3)
    conv4 = resnet3.residual_block(pool3,4*BASE)
    deconv1 = deconv_batch(conv4,4*BASE)
    res1 = resnet3.residual_block(deconv1,nb_filter=4*BASE)
    res1 = merge([res1,conv3],mode='concat')
    deconv2 = deconv_batch(res1,2*BASE)
    res2 = resnet3.residual_block(deconv2,nb_filter=2*BASE)
    res2 = merge([res2,conv2],mode='concat')
    deconv3 = deconv_batch(res2,BASE)
    res3 = resnet3.residual_block(deconv3,nb_filter=BASE)
    res3 = merge([res3,conv1],mode='concat')
    predictions1 = Convolution3D(1,1,1,1,activation='sigmoid',name='l1')(res1)
    predictions2 = Convolution3D(1,1,1,1,activation='sigmoid',name='l2')(res2)
    predictions3 = Convolution3D(1,1,1,1,activation='sigmoid',name='l3')(res3)
 
    model = Model(input=[inputs],output=[predictions1,predictions2,predictions3])

    model.compile(optimizer=Adam(lr=0.0005),loss=resnet3.dice_coef_loss,loss_weights=[0.2,0.3,0.5])
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
    data,labels,pos = preprocess.load_luna()
    N_VAL = 100
    data_t = data[:-N_VAL]
    labels_t = labels[:-N_VAL]
    pos_t = pos[:-N_VAL]
    data_v = data[-N_VAL:]
    labels_v = labels[-N_VAL:]
    pos_v = pos[-N_VAL:]

    checkpoint = ModelCheckpoint('logs/wegihts.{epoch:02d}--{val_loss:.2f}.hdf5')
    save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)


    BATCH_SIZE = 24
    train_generator= preprocess.generate_luna_batch(data_t,labels_t,pos_t,batch_size=BATCH_SIZE)
    valid_generator= preprocess.generate_luna_batch(data_v,labels_v,pos_v,batch_size=1)

    #exit()
    TRAIN = True
    if TRAIN:
        model.fit_generator(generator= train_generator,validation_data=valid_generator,
                nb_val_samples=400,nb_epoch=35,samples_per_epoch=4000,callbacks=[checkpoint,save_best],nb_worker=4)


    model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
    test_generator= preprocess.generate_luna_batch(data_v,labels_v,pos_v,batch_size=1)

    data_test=[]
    mask_test=[]
    for _ in range(20):
        x,y=next(test_generator)
        data_test.append(x['inputs'])
        mask_test.append(y['l3'])

    data_test = np.asarray(data_test)
    mask_test = np.asarray(mask_test)
    data_test = np.squeeze(data_test,axis=1)
    mask_test = np.squeeze(mask_test,axis=1)
    predictions=model.predict(data_test,batch_size=1)
    predict=predictions[2]
    for i in range(len(data_test)):
        print "nodule pixels:",np.sum(mask_test[i,...])
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
