import glob
import numpy as np
import os
import SimpleITK as sitk
import skimage.transform
import scipy.ndimage
import pandas as pd
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
import time

from joblib import Parallel, delayed
import sys
sys.path.append('/home/ronens1/DSB3/src')

import preprocess1 as pre

LIDC_ROOT = '/mnt/storage/forShai/LIDC/'
PROCESSED_DIR = '/home/ronens1/lidc/processed/'

NEW_SPACING = pre.NEW_SPACING



def draw_circles(image,cands,spacing):

    #make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape,dtype='uint8')

    #run over all the nodules in the lungs
    voxel_cands=[]
    for ca in cands.values:

        radius = ca[4]/2.
        radius_voxel = np.array((radius/spacing[0],radius/spacing[1],radius/spacing[2]))
        center_coord = np.array(ca[1:4])/spacing
        voxel_cands.append(np.concatenate((center_coord,radius_voxel)))
        #print image.shape,ca
        diameter_voxels = np.ceil(2*np.array([radius,radius,radius])/spacing).astype(int)
        corner1 = np.maximum((center_coord - diameter_voxels/2.-(1,1,1)).astype(int),(0,0,0))
        corner2 = np.minimum((center_coord+diameter_voxels/2.+(1,1,1)).astype(int),image.shape)
        x = np.arange(corner1[0],corner2[0])-center_coord[0]
        y = np.arange(corner1[1],corner2[1])-center_coord[1]
        z = np.arange(corner1[2],corner2[2])-center_coord[2]

        xx,yy,zz = np.meshgrid(x,y,z)
        nodule = (xx/radius_voxel[0])**2 + (yy/radius_voxel[1])**2 + (zz/radius_voxel[2])**2 <= 1
        image_mask[corner1[0]:corner2[0],
                corner1[1]:corner2[1],
                corner1[2]:corner2[2]] = nodule.astype(int)

    return image_mask,voxel_cands


def crop_it(image,label,voxel_cads):
    margin  = 32
    if np.sum(label)>0:
        nz = np.nonzero(label)
        x_min =np.min(nz[0])-margin
        x_max = np.max(nz[0])+margin
        y_min = np.min(nz[1])-margin
        y_max = np.max(nz[1])+margin
        z_min = np.min(nz[2])-margin
        z_max = np.max(nz[2])+margin
    else:
        x_min = label.shape[0]/4
        x_max = (3*label.shape[0])/4
        y_min = label.shape[1]/4
        y_max = (3*label.shape[1])/4
        z_min = label.shape[2]/4
        z_max = (3*label.shape[2])/4

    x_min = max(x_min,0)
    y_min = max(y_min,0)
    z_min = max(z_min,0)
    x_max = min(x_max,label.shape[0]-1)
    y_max = min(y_max,label.shape[1]-1)
    z_max = min(z_max,label.shape[2]-1)
    image1 = image[x_min:x_max,y_min:y_max,z_min:z_max]
    label1 = label[x_min:x_max,y_min:y_max,z_min:z_max]
    for i,x in enumerate(voxel_cads):
        voxel_cads[i][0:3] = x[0:3] - np.array((x_min,y_min,z_min))
    print "cropped",image1.shape
    return image1,label1,voxel_cads




def create_slices(imagePath,cads):
    img,spacing,direction = pre.process_image(imagePath,info=True)
    #pre.plot_image(img)
    imageName = os.path.split(imagePath)[-1]
    image_cads = cads[cads.iloc[:,0] == imageName]

    nodule_mask,voxel_cads = draw_circles(img,image_cads,NEW_SPACING)

    DEBUG = False
    if DEBUG: # and direction[0]<0:
        big = 0
        ind1 = 0
        ind2 = 0
        ind3 = 0
        for k in range(nodule_mask.shape[0]):
            b = np.sum(nodule_mask[k,:,:])
            if b >= big:
                ind1 = k
                big = b
        big= 0
        for k in range(nodule_mask.shape[1]):
            b = np.sum(nodule_mask[:,k,:])
            if b >= big:
                ind2 = k
                big = b
        big = 0
        for k in range(nodule_mask.shape[2]):
            b = np.sum(nodule_mask[:,:,k])
            if b >= big:
                ind3 = k
                big = b
        print imageName,"index",ind1,ind2,ind3,"mask pixels:",np.sum(nodule_mask)
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(np.squeeze(img[ind1,(ind2-20):(ind2+20),(ind3-20):(ind3+20)]),cmap=plt.cm.gray)
        plt.subplot(212)
        plt.imshow(np.squeeze(nodule_mask[ind1,(ind2-20):(ind2+20),(ind3-20):(ind3+20)]),cmap=plt.cm.gray)
        plt.figure(2)
        plt.subplot(211)
        plt.imshow(np.squeeze(img[(ind1-20):(ind1+20),ind2,(ind3-20):(ind3+20)]),cmap=plt.cm.gray)
        plt.subplot(212)
        plt.imshow(np.squeeze(nodule_mask[(ind1-20):(ind1+20),ind2,(ind3-20):(ind3+20)]),cmap=plt.cm.gray)

        plt.figure(3)
        plt.subplot(211)
        plt.imshow(np.squeeze(img[(ind1-20):(ind1+20),(ind2-20):(ind2+20),ind3]),cmap=plt.cm.gray)
        plt.subplot(212)
        plt.imshow(np.squeeze(nodule_mask[(ind1-20):(ind1+20),(ind2-20):(ind2+20),ind3]),cmap=plt.cm.gray)
        plt.show()

    #img,nodule_mask,voxel_cads = crop_it(img,nodule_mask,voxel_cads)
    img,corner1,corner2 = pre.get_lung_region(img)
    PLOT = False
    if PLOT:
        print imagePath
        pre.plot_image(img)
    nodule_mask = nodule_mask[corner1[0]:corner2[0],corner1[1]:corner2[1],corner1[2]:corner2[2]]
    if img.shape[0]<40 or img.shape[1]<40 or img.shape[2]<40:
        print "Error?",imagePath,img.shape
    np.save(PROCESSED_DIR+imageName+'.npy',img)
    np.save(PROCESSED_DIR+imageName+'_nodules.npy',nodule_mask)
if __name__ == "__main__":
    print('Usage: %s LIDC_ROOT/ PROCESSED_DIR/' % sys.argv[0])
    if len(sys.argv) == 3:
        LIDC_ROOT = sys.argv[1]
        PROCESSED_DIR = sys.argv[2]
    CSV_DIR = os.path.dirname(PROCESSED_DIR[:-1])+'/csv/'

    uids = pd.read_csv(CSV_DIR + "list.csv")
    names  = uids.ix[:,0].values.tolist()
    paths = []
    #names=['1.3.6.1.4.1.14519.5.2.1.6279.6001.199220738144407033276946096708']
    #names = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.234400932423244218697302970157']
    #names= ['1.3.6.1.4.1.14519.5.2.1.6279.6001.281254424459536762125243157973']
    #names = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.130438550890816550994739120843',
    #        '1.3.6.1.4.1.14519.5.2.1.6279.6001.398955972049286139436103068984',
    #        '1.3.6.1.4.1.14519.5.2.1.6279.6001.208737629504245244513001631764',
    #        '1.3.6.1.4.1.14519.5.2.1.6279.6001.178391668569567816549737454720',
    #        '1.3.6.1.4.1.14519.5.2.1.6279.6001.503980049263254396021509831276']

    for name in names:
        paths.append(glob.glob(LIDC_ROOT+'*/*/'+name)[0])
    #print imagePaths
    cads = pd.read_csv(CSV_DIR + "annoations.csv")
    Parallel(n_jobs=-1,verbose=1)(delayed(create_slices)(imagePath, cads) for imagePath in paths)
