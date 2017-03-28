PROCESSED_IMAGES_ROOT ='../data/processed/'


from keras.models import Model
from keras.layers import Activation,Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.objectives import binary_crossentropy,mse,mae
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from deconv3D import Deconvolution3D
import preprocess
import numpy as np
import resnet3
import keras.backend as K
import tensorflow as tf
import glob
from scipy import ndimage

from matplotlib import pyplot as plt


BEST_WEIGHTS_PATH = 'logs/best_weights_seg1.hdf5' 



def pos_loss(y_true,y_pred):
    l1 = mse(y_true[:,0:2],y_pred[:,0:2])
    l2 = mae(y_true[:,3],y_pred[:,3])
    l3 = binary_crossentropy(y_true[:,4],y_pred[:,4])
    return (l1 + l2)*y_true + l3

def narrow(prev,channels,kernel=3,stride=1,activation='relu',layers=3):
    conv = prev
    for _ in range(layers/2):
        conv = conv_batch(conv,channels,kernel=kernel,activation=activation)
        conv = conv_batch(conv,channels/2,kernel=1,activation=None)
    conv = conv_batch(conv,channels,kernel=kernel,stride=stride,activation=activation)
    return conv
 
def conv_batch(prev,channels,kernel=3,stride=1,activation='relu'):
    conv = Convolution3D(channels,kernel,kernel,kernel,subsample=(stride,stride,stride),init='he_normal',border_mode='same')(prev)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)    
    return conv
def deconv_batch(prev,channels,kernel=3,stride=2,activation='relu'):
    shape = prev._keras_shape
    output_shape = (None,2*shape[1],2*shape[2],2*shape[3],channels)
    deconv = Deconvolution3D(channels,kernel,kernel,kernel,output_shape=output_shape,subsample=(stride,stride,stride),init='he_normal',border_mode='same')(prev)
    deconv = BatchNormalization()(deconv)
    deconv = Activation('relu')(deconv)

    return deconv

def get_model(input_shape=(preprocess.SEG_CROP,preprocess.SEG_CROP,preprocess.SEG_CROP,1)):
    BASE = 8
    resnet3.handle_dim_ordering()
 
    inputs = Input(shape=input_shape,dtype='float32',name='inputs')
    conv1 = conv_batch(inputs,BASE)
    pool1 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv1)
    conv2 = conv_batch(pool1,2*BASE)
    pool2 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv2)
 
    conv3 = narrow(pool2,4*BASE) 
    pool3 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv3)
 
    conv4 = conv_batch(pool3,8*BASE)
    deconv1 = deconv_batch(conv4,4*BASE)
    conv5 = narrow(deconv1,4*BASE)
    merge1 = merge([conv5,conv3],mode='concat')
    deconv2 = deconv_batch(merge1,2*BASE)
    conv6 = narrow(deconv2,2*BASE)
    merge2 = merge([conv6,conv2],mode='concat')
    deconv3 = deconv_batch(merge2,BASE)
    conv7 = narrow(deconv3,BASE)
    merge3 = merge([conv7,conv1],mode='concat')
    predictions1 = Convolution3D(1,1,1,1,activation='sigmoid',name='l1',init='normal')(merge1)
    predictions2 = Convolution3D(1,1,1,1,activation='sigmoid',name='l2',init='normal')(merge2)
    predictions3 = Convolution3D(1,1,1,1,activation='sigmoid',name='l3',init='normal')(merge3)
 
    # position part
    #pool4 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(conv4)
    #pos_conv1 = conv_batch(pool4,16*BASE)
    #pool5 = MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2))(pos_conv1)
    #pool5 = Flatten()(pool5)
    #pos_target = Dense(4,activation='sigmoid',init='normal',border_mode='same',name='pos')(pool5)
    

    #model = Model(input=[inputs],output=[predictions1,predictions2,predictions3,pos_target])
    model = Model(input=[inputs],output=[predictions1,predictions2,predictions3])
  
    adam  = Adam(lr=0.001)
    #loss ={'l1': binary_crossentropy,'l2': binary_crossentropy,'l3': binary_crossentropy,'pos': pos_loss}  
    #loss_weights ={'l1': 0.2 ,'l2': 0.3,'l3':0.5,'pos': 1.0}  
    loss = binary_crossentropy  
    loss_weights ={'l1': 0.2 ,'l2': 0.3,'l3':0.5}  
 

    #model.compile(optimizer=adam,loss=loss,metrics=[resnet3.dice_coef_loss],loss_weights=loss_weights)
    model.compile(optimizer=adam,loss=resnet3.dice_coef_loss,loss_weights=loss_weights)
  
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
    DISK = False
    if DISK:
        files = glob.glob(preprocess.SEG_ROOT+'*') # image file names
        data = [x for x in files if not 'nodule' in x]
        print "# image files:", len(data)
        labels = [None]*len(data)
    else:
        data,labels,positions = preprocess.load_luna()
    N_VAL = 80
    data_t = data[:-N_VAL]
    labels_t = labels[:-N_VAL]
    positions_t = positions[:-N_VAL] 
    data_v = data[-N_VAL:]
    labels_v = labels[-N_VAL:]
    positions_v = positions[-N_VAL:]

    checkpoint = ModelCheckpoint('logs/wegihts.{epoch:02d}--{val_loss:.2f}.hdf5')
    save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)


    BATCH_SIZE = 10
    #BATCH_SIZE = 16
    train_generator= preprocess.generate_luna_batch(data_t,labels_t,positions_t,batch_size=BATCH_SIZE,disk=DISK)
    valid_generator= preprocess.generate_luna_batch(data_v,labels_v,positions_v,batch_size=1,disk=DISK)

    #exit()
    #class_weight={0: 1., 1: 1.}
    TRAIN = True
    if TRAIN:
        model.fit_generator(generator= train_generator,validation_data=valid_generator,
                nb_val_samples=400,nb_epoch=30,samples_per_epoch=4000,callbacks=[checkpoint,save_best],
                nb_worker=1,verbose=1)


    model.load_weights(BEST_WEIGHTS_PATH,by_name=True)
    test_generator= preprocess.generate_luna_batch(data_v,labels_v,positions_v,batch_size=1,disk=DISK)

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
        print "nodule pixels:",np.sum(mask_test[i,...]),'max prob',np.max(predict[i,...])
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
