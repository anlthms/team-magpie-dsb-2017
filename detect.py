from keras.models import Model
from keras.layers import Input, Dense, Convolution3D,Flatten, BatchNormalization, Dropout, Merge,merge
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import preprocess1 as pre
import numpy as np
import sys
import os
import resnet3
import keras.backend as K
import tensorflow as tf
import seg_lidc as seg
import resnet3
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
import skimage.morphology 
import skimage.measure
from tqdm import tqdm

PAD = 20
def get_candidates(image,predict,nmax=5):
        thresh=0.2
        candidates = []
        mask = (predict>thresh).astype(int)
        for _ in range(3):
            mask = skimage.morphology.binary_opening(mask)

        for _ in range(5):
            mask = skimage.morphology.binary_dilation(mask)
        for _ in range(5):
            mask = skimage.morphology.binary_erosion(mask)


        labels,num = skimage.morphology.label(mask,return_num=True)
        print "# nodules",num, "# pixels:",np.sum(mask)
        volumes=[]
        for l in range(1,num+1):
            volumes.append(np.sum(labels==l))
        order = np.argsort(volumes)[::-1]
        for l in range(0,min(nmax,num)):
            pix = np.where(labels == order[l]+1)
            #print "vol",len(pix[0])
            cent = (int(np.mean(pix[0])),int(np.mean(pix[1])),int(np.mean(pix[2])))
            candidates.append(cent)
        return candidates

def visualize(image,predict):
        thresh=0.3
        mask = (predict>thresh).astype(int)
        for _ in range(3):
            mask = skimage.morphology.binary_dilation(mask)
        labels,num = skimage.morphology.label(mask,return_num=True)
        print "# nodules",num
        print "nodule pixels:",np.sum(mask),'max prob',np.max(predict)
        for l in range(1,num+1):
            pix = np.where(labels == l)
            cent = (np.mean(pix[0]),np.mean(pix[1]),np.mean(pix[2]))
            print l,cent,len(pix[0])
        smooth = ndimage.gaussian_filter(predict, sigma=(5, 5, 5), order=0)
        cand_pos = np.unravel_index(np.argmax(smooth),smooth.shape)
        crop_image = pre.crop(image,cand_pos) 
        crop_predict = pre.crop(predict,cand_pos)
        crop_size = crop_image.shape
        plt.figure(1)
        plt.subplot(321)
        plt.imshow(np.squeeze(255*crop_image[crop_size[0]/2,:,:]),cmap=plt.cm.gray)
        plt.subplot(322)
        plt.imshow(np.squeeze(255*crop_predict[crop_size[0]/2,:,:]),cmap=plt.cm.gray)
        plt.subplot(323)
        plt.imshow(np.squeeze(255*crop_image[:,crop_size[1]/2,:]),cmap=plt.cm.gray)
        plt.subplot(324)
        plt.imshow(np.squeeze(255*crop_predict[:,crop_size[1]/2,:]),cmap=plt.cm.gray)
        plt.subplot(325)
        plt.imshow(np.squeeze(255*crop_image[:,:,crop_size[2]/2]),cmap=plt.cm.gray)
        plt.subplot(326)
        plt.imshow(np.squeeze(255*crop_predict[:,:,crop_size[2]/2]),cmap=plt.cm.gray)
        plt.show()


def get_model(patch=False,refined=False):
    if patch:
	PART_SHAPE = pre.SEG_CROP
    else:
        PART_SHAPE = [pre.NEW_SHAPE[0],pre.NEW_SHAPE[1],pre.NEW_SHAPE[2]/2+PAD]
    model = seg.get_model(PART_SHAPE+[1])
    
    if refined:
        model.load_weights(seg.REFINE_MODEL,by_name=True)
    else:
        model.load_weights(seg.BEST_WEIGHTS_PATH,by_name=True)

    model.compile(optimizer='adam',loss='binary_crossentropy')
    return model

def get_activations(model, X_batch):
    get_activations = K.function([model.layers[0].input,K.learning_phase()], [model.get_layer(name='pre-detect').output])
    get_detections = K.function([model.layers[0].input,K.learning_phase()], [model.get_layer(name='detect').output])
 
    activations = get_activations([X_batch,0])
    detections = get_detections([X_batch,0])
    features = np.concatenate((activations,detections),axis=-1)
    return features

REFINED = True
#DataSet =  'LIDC'
DataSet = 'DSB'
Mode = 'test'
print('Usage: %s DataSet <REFINED 1/0> SEG_ROOT/ ORIGINAL_IMAGES_ROOT/ PROCESSED_IMAGES_ROOT/ LABELS_FILE Mode' % sys.argv[0])
if len(sys.argv) == 8:
    DataSet = sys.argv[1]
    REFINED = (sys.argv[2] == '1')
    pre.SEG_ROOT = sys.argv[3]
    pre.ORIGINAL_IMAGES_ROOT = sys.argv[4]
    pre.PROCESSED_IMAGES_ROOT = sys.argv[5]
    pre.LABELS_FILE = sys.argv[6]
    Mode = sys.argv[7]

print('DataSet %s REFINED %s' % (DataSet, REFINED))


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
K.set_session(session)
model = get_model(refined=REFINED)  
model_patch = get_model(patch=True,refined=REFINED)
print "loading data..."
if DataSet == 'DSB':
    print "DSB"
    data,labels,names = pre.load_numpy_images(dataset=Mode)
    print len(data)
    save_dir = pre.DETECTIONS_DSB_ROOT+Mode+'/'
    nmax = 5
else:
    print "LIDC"
    data,labels = pre.load_lidc()
    save_dir = pre.DETECTIONS_LIDC_ROOT
    nmax = 10

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print "predicting..."
#data = data[:4]
data_test=[]
VIS = False
candidates = []
for i in range(len(data)):
    #print data[i].shape
    image =  pre.pad(data[i]).astype(np.float32)/pre.MAX_PIXEL
    data_test = np.expand_dims(np.expand_dims(image,4),0)
    #print image.shape
    part= [[0,pre.NEW_SHAPE[2]/2+PAD],[pre.NEW_SHAPE[2]/2-PAD,pre.NEW_SHAPE[2]]]
    sub = [[0,pre.NEW_SHAPE[2]/2],[PAD,pre.NEW_SHAPE[2]]]
    prediction = np.zeros(np.array(image.shape))
    features = np.zeros([image.shape[0]/48,image.shape[1]/48,image.shape[2]/48,151])
    split = prediction.shape[2]/96
    for k in range(2):
        part_image = data_test[:,:,:,part[k][0]:part[k][1],:]
        part_predicted = np.squeeze(model.predict(part_image,batch_size=1,verbose=1)[2])
        part_features = np.squeeze(get_activations(model,part_image)[0])
        prediction[:,:,k*prediction.shape[2]/2:(k+1)*prediction.shape[2]/2] = part_predicted[:,:,sub[k][0]:sub[k][1]]
        if k == 0:
            features[:,:,0:split,:] = part_features[:,:,0:split,:]
        else:
            features[:,:,split:,:] = part_features[:,:,(split+part_features.shape[2]-features.shape[2]):,:]
    if DataSet == 'LIDC':
        true = pre.pad(labels[i])
    else:
        true = np.zeros(image.shape)
    if VIS:
        visualize(image,prediction)
    crop_shape = np.array(pre.SEG_CROP)
    cuts = np.zeros([nmax]+crop_shape.tolist(),dtype=np.uint16)
    cands = get_candidates(image,prediction,nmax)
    candidates.append(cands)
    if DataSet == 'LIDC': 
        correct = 0
        for j,cand in enumerate(cands):
            true_crop = pre.crop(true,cand)
            if np.sum(true_crop[crop_shape[0]/4:3*crop_shape[0]/4,
                crop_shape[1]/4:3*crop_shape[1]/4,
                crop_shape[2]/4:3*crop_shape[2]/4])>0:
                correct += 1 
        ls,num = skimage.measure.label(true,return_num=True)        
       
        print "correct",correct,"out of",float(len(cands)),"actual",num
    else:
        X=np.zeros([nmax]+crop_shape.tolist()+[1])
        for j,cand in enumerate(cands):
            cand_crop = pre.crop(image,cand)
            cuts[j,...]=(cand_crop*pre.MAX_PIXEL).astype(np.uint16)
            X[j,:,:,:,0] = cand_crop
        activations=get_activations(model_patch,X)[0]
        np.save(save_dir+'detection_'+names[i]+'.npy',cuts)
        np.save(save_dir+'activation_'+names[i]+'.npy',activations)
        np.save(save_dir+'features_'+names[i]+'.npy',features) 
np.save(save_dir+'detections.npy',np.asarray(candidates))
    

     

