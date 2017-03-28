PROCESSED_IMAGES_ROOT ='../data/processed/'

from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge,Activation
from keras.layers import AveragePooling3D, MaxPooling3D, Cropping3D
from keras.callbacks import ModelCheckpoint
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.optimizers import Adam,RMSprop
from keras.regularizers import l2
from deconv3D import Deconvolution3D
import SegDenseNet3 as segnet
import preprocess1 as preprocess
import numpy as np
import keras.backend as K
import tensorflow as tf
import metrics

from matplotlib import pyplot as plt


BEST_WEIGHTS_PATH = 'models/best_seg_lidc.hdf5' 
REFINE_MODEL = 'models/refine_lidc.hdf5' 

DETECTIONS_PATH = preprocess.DETECTIONS_LIDC_ROOT+'detections.npy'
OUT = 1
cost = binary_crossentropy
#cost2 = metrics.dice_coef_loss

def get_model(input_shape=preprocess.SEG_CROP+[1]):
    net = segnet.Network(input_shape = input_shape,
            n_filters_first_conv = 20,
            n_pool = 3,
            growth_rate = 8,
            n_layers_per_block = 3,
            dropout_p = 0.1)
    net.model.compile(optimizer=Adam(lr=0.001),loss=cost)

    net.model.summary()
    return net.model


def get_model2(input_shape=preprocess.SEG_CROP+[1]):
    net = segnet.Network(input_shape = input_shape,
            n_filters_first_conv = 16,
            n_pool = 3,
            growth_rate = 10,
            n_layers_per_block = 3,
            dropout_p = 0.1)
    net.model.compile(optimizer=Adam(lr=0.001),loss=cost)

    net.model.summary()
    return net.model


def get_model3(input_shape=preprocess.SEG_CROP+[1]):
    net = segnet.Network(input_shape = input_shape,
            n_filters_first_conv = 20,
            n_pool = 3,
            growth_rate = 16,
            n_layers_per_block = 3,
            dropout_p = 0.2)
    net.model.compile(optimizer=Adam(lr=0.001),loss=cost)

    net.model.summary()
    return net.model




def main():
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    K.set_session(session)

    RESUME = False
    if RESUME:
        model = load_model(BEST_MODEL_PATH)
    else:
        model = get_model2()   

    REFINE= False
    print "loading data..."
    data,labels = preprocess.load_lidc()
    N_VAL = 100
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
        model.compile(optimizer=Adam(lr=0.0002),loss = binary_crossentropy)
 
    else:
        save_best = ModelCheckpoint(BEST_WEIGHTS_PATH,save_best_only=True)

    BATCH_SIZE = 10
    train_generator= preprocess.generate_lidc_batch(data_t,labels_t,detections=detections_t,batch_size=BATCH_SIZE,out=OUT)
    valid_generator= preprocess.generate_lidc_batch(data_v,labels_v,detections=detections_v,batch_size=BATCH_SIZE,out=OUT)

    TRAIN = True

    if TRAIN:
        model.fit_generator(generator= train_generator,validation_data=valid_generator,
                nb_val_samples=400,nb_epoch=60,samples_per_epoch=4000,callbacks=[save_best],nb_worker=1) # nb_epoch = 40
	if not REFINE:
		model.compile(optimizer=Adam(lr=0.0001),loss=cost)
		train_generator= preprocess.generate_lidc_batch(data_t,labels_t,detections=detections_t,batch_size=BATCH_SIZE,neg_fraction=0.8,out=OUT)
		valid_generator= preprocess.generate_lidc_batch(data_v,labels_v,detections=detections_v,batch_size=1,neg_fraction=0.8,out=OUT)
		model.fit_generator(generator= train_generator,validation_data=valid_generator, nb_val_samples=400,nb_epoch=4,samples_per_epoch=4000,callbacks=[save_best],nb_worker=1)

    if REFINE:
        model.load_weights(REFINE_MODEL,by_name=True)
    else:
        model.load_weights(BEST_WEIGHTS_PATH,by_name=True)

    test_generator= preprocess.generate_lidc_batch(data_v,labels_v,detections=detections_v,batch_size=1,out=OUT)

    data_test=[]
    mask_test=[]
    for _ in range(64):
        x,y=next(test_generator)
        data_test.append(x['inputs'])
        mask_test.append(y['l3'])

    data_test = np.concatenate(data_test,axis=0)
    mask_test = np.concatenate(mask_test,axis=0)
    predictions=model.predict(data_test,batch_size=BATCH_SIZE)
    predict=predictions
    print predict.shape
    print "Test",data_test.shape,predict.shape
    for i in range(len(data_test)):
        pix_sum = np.sum(mask_test[i,...])
        pix_max = np.max(predict[i,...])
        shape = data_v[i].shape
        print "nodule pixels:",pix_sum,pix_max,shape
        PLOT = True
        thresh = 0.3
        if PLOT and ((pix_sum>0 and pix_max<thresh) or (pix_sum==0 and pix_max>thresh)) :
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
