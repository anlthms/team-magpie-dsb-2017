import dicom
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from joblib import Parallel, delayed
import pandas as pd
import glob
import random
from tqdm import tqdm
import threading
import sys
from skimage import measure,morphology

ORIGINAL_IMAGES_ROOT='../data/stage1/'
PROCESSED_IMAGES_ROOT ='../data/processed/'
DETECTIONS_DSB_ROOT = '../data/detections_dsb/'
DETECTIONS_LIDC_ROOT = '../data/detections_lidc/'
LABELS_FILE = '../data/stage1_labels.csv'
SEG_ROOT = '/home/ronens1/lidc/processed/'

#NEW_SHAPE=[176,168,152] # for spacing [2,2,2]
#NEW_SHAPE =[232,224,200] # for spacing [1.5,1.5,1.5,]

NEW_SHAPE = [272,264,232] 
#NEW_SHAPE = [268,260,228]
NEW_SPACING = [1.3,1.3,1.3] # where last index is in leg-head direction

SEG_CROP = [48,48,48]
#SEG_CROP = [52,52,52]
MAX_PIXEL = 65535
    
#det_crop = (31,30,27)
det_crop = (5,5,4)
random.seed(2)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def seg_crop (image,corner1,crop_size):
    corner2=np.array(corner1)+np.array(crop_size)
    return image[corner1[0]:corner2[0],corner1[1]:corner2[1],corner1[2]:corner2[2]]
def random_flip(image,label):
        flip = np.random.randint(0,2,3)
        if flip[0]:
                image = image[::-1,:,:]
		label = label[::-1,:,:]	
        if flip[1]:
                image = image[:,::-1,:]
		label = label[:,::-1,:]
        if flip[2]:
                image = image[:,:,::-1]
		label = label[:,:,::-1]
        return image,label,flip



def crop(image,position,pad=0):
    crop_shape = np.array(SEG_CROP)+np.array([2*pad,2*pad,2*pad])
    corner1 = np.array(position)-crop_shape/2
    corner1s = np.maximum(corner1, np.array([0,0,0]))
    #corner2 = corner1+crop_shape
    corner2 = np.array(position)+crop_shape/2
   
    corner2s = np.minimum(corner2,np.array(image.shape))
    place1= corner1s-corner1
    place2= crop_shape+corner2s-corner2
    #corner1 = corner2-crop_shape
    patch = np.zeros(crop_shape)
    patch[place1[0]:place2[0],place1[1]:place2[1],place1[2]:place2[2]] =  image[corner1s[0]:corner2s[0],corner1s[1]:corner2s[1],corner1s[2]:corner2s[2]]

    if np.any(patch.shape != crop_shape):
        print "crop error",image.shape,patch.shape,crop_shape
    return patch

def  augment1(patch):
    if random.random() < 0.5:
        patch = patch[::-1,:,:,:]
    if random.random() < 0.5:
        patch = patch[:,::-1,:,:]
    #if random.random() < 0.5:
    #    patch = patch[:,:,::-1,:]
    perm = [0,1]
    #perm = [0,1,2]
    random.shuffle(perm)
    perm = perm+[2,3]
    #perm = perm + [3]
    patch = np.transpose(patch,perm)
    return patch


def augment(patch,label):
    if random.random() < 0.5:
        patch = patch[::-1,:,:]
	label = label[::-1,:,:]
    if random.random() < 0.5:
        patch = patch[:,::-1,:]
	label = label[:,::-1,:]
    if random.random() < 0.5:
        patch = patch[:,:,::-1]
	label = label[:,:,::-1]
    #perm = [0,1]
    perm = [0,1,2]
    random.shuffle(perm)
    #perm = perm+[2]
    patch = np.transpose(patch,perm)
    label = np.transpose(label,perm)
    return patch,label



def seg_downsample(image,mode='simple'):

    if mode == 'simple':
        return image[::2,::2,::2]
    else:

        c1 = image[::2,::2,::2,:]
        c2 = image[::2,::2,1::2,:]
        c3 = image[::2,1::2,::2,:]
        c4 = image[::2,1::2,1::2,:]
        c5 = image[1::2,::2,::2,:]
        c6 = image[1::2,::2,1::2,:]
        c7 = image[1::2,1::2,::2,:]
        c8 = image[1::2, 1::2, 1::2,:]
        c =np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8),axis=3)
        return np.amax(c,axis=3,keepdims=True)




def load_lidc():
    print('loading from', SEG_ROOT)
    filenames = glob.glob(SEG_ROOT+'*')
    if os.path.exists('quick-mode'):
        np.random.shuffle(filenames)
        filenames = filenames[:int(len(filenames) * 0.1)]
        print('Quick mode - reading %d files' % (len(filenames)))

    data = []
    labels = []
    positions = []
    print 'loading seg data'
    for name in tqdm(filenames):
        if  'nodules' in name :
            continue
        image = crop_box(np.load(name))
        if image.shape[0]<40:
            print "Errro?",image.shape
        data.append(image)
        nodules_name = os.path.splitext(name)[0]+'_nodules.npy'
        mask = crop_box(np.load(nodules_name))
        labels.append(mask)
    max_shape = np.max(np.array([x.shape for x in data]),axis=0)
    print "Max shape",max_shape
    combined =  list(zip(data,labels))
    random.shuffle(combined)
    data[:],labels[:]=zip(*combined)
    return data,labels

def random_flip1(image,flip_axis=(0,1,2)):
	if 0 in flip_axis and random.random() < 0.5:
                image = image[::-1,:,:]
        if 1 in flip_axis and random.random() < 0.5:
                image = image[:,::-1,:]
        if 2 in flip_axis and random.random() < 0.5:
                image = image[:,:,::-1]
        return image

def random_permute(image,label):
        p = np.random.permutation(3)
        image = np.transpose(image,p)
        label = np.transpose(label,p)
        return image,label,p


def get_lung_region(image):
    mask = np.array(image > 0.4*MAX_PIXEL,dtype=np.int8)
    for k in range(4):
        mask = morphology.binary_opening(mask)
    #for k in range(4):
    #   mask = morphology.binary_erosion(mask)
    #for k in range(4):
    #   mask = morphology.binary_dilation(mask)

    mask = mask+1
    for i in range(image.shape[2]):
        mask_slice = np.squeeze(mask[:,:,i])
        labels = measure.label(mask_slice)
        #ackground_labels = [labels[0,0],labels[0,mask_slice.shape[1]-1],
        #       labels[mask_slice.shape[0]-1,mask_slice.shape[1]-1],labels[mask_slice.shape[0]-1,0]]
        background_labels = np.unique(labels[0,:].tolist()+labels[-1,:].tolist()+labels[:,0].tolist()+labels[:,-1].tolist())
        for b in background_labels:
            mask_slice[ b == labels ] = 2
        mask_slice = 2 - mask_slice
        thresh = 6400./(NEW_SPACING[0]*NEW_SPACING[1]) 
        DEBUG = False
        if DEBUG:
            print i,np.sum(mask_slice)
            if i % 10 == 0:
                plt.subplot(211)
                plt.imshow(image[:,:,i])
                plt.subplot(212)
                plt.imshow(mask_slice)
                plt.show()
        if np.sum(mask_slice) < thresh:
            mask[:,:,i]=np.zeros(mask_slice.shape)
        else:
            mask[:,:,i] = mask_slice
    for k in range(4):
       mask = morphology.binary_erosion(mask)
    for k in range(4):
       mask = morphology.binary_dilation(mask)

    proj0 = np.max(mask,axis=(1,2))
    proj1 = np.max(mask,axis=(0,2))
    proj2 = np.max(mask,axis=(0,1))

    #for k in range(8):
    #    proj2 = morphology.binary_dilation(proj2)
    #ls,num = measure.label(proj2,return_num=True)
    #length = np.zeros(num+1)
    #for l in range(1,num+1):
    #    length[l] = np.sum(proj2[ls==l])
    #ind = np.argsort(length)
    #proj2[ls != ind[-1]] = 0

    a0 = np.argmax(proj0)
    b0 = len(proj0)-np.argmax(proj0[::-1])-1
    a1 = np.argmax(proj1)
    b1 = len(proj1)-np.argmax(proj1[::-1])-1
    a2 = np.argmax(proj2)
    b2 = len(proj2)-np.argmax(proj2[::-1])-1
    corner1 = np.array([a0,a1,a2])
    corner2 = np.array([b0,b1,b2])
    corner1 = np.maximum(corner1-np.array([6,6,6]),np.array([0,0,0]))
    corner2 = np.minimum(corner2+np.array([6,6,6]),np.array(image.shape))
    croped = image[corner1[0]:corner2[0],corner1[1]:corner2[1],corner1[2]:corner2[2]]
    print "lung",image.shape,corner1,corner2,croped.shape
    #if image.shape[2]>220:
    #    print image.shape
    #    print proj2
    return croped,corner1,corner2


def crop_box(image):
    if np.any(np.array(image.shape)>np.array(NEW_SHAPE)):
        #print image.shape
        corner1 = np.maximum((np.array(image.shape)-np.array(NEW_SHAPE))/2,np.array([0,0,0]))
        corner2 = corner1+ np.array(NEW_SHAPE)
        image = image[corner1[0]:corner2[0],corner1[1]:corner2[1],corner1[2]:corner2[2]]
        #print "->",image.shape
    return image

def pad(image,canvas=None):
    if canvas is None:
        canvas = np.zeros(NEW_SHAPE,dtype=np.uint16)
    else:
        canvas[:,:,:]=0
    a0 = (NEW_SHAPE[0]-image.shape[0])/2
    b0 = (NEW_SHAPE[1]-image.shape[1])/2
    c0 = (NEW_SHAPE[2]-image.shape[2])/2
    canvas[a0:a0+image.shape[0],b0:b0+image.shape[1],c0:c0+image.shape[2]]= image
    return canvas

def find_nodule(image,label):
    nz = np.nonzero(label)
    xs = nz[0]
    ys = nz[1]
    zs  = nz[2]
    npix = len(xs)
    choose =  random.randint(0,npix-1)
    x = xs[choose] + random.randint(-SEG_CROP[0]/4,SEG_CROP[0]/4)
    y = ys[choose] + random.randint(-SEG_CROP[1]/4,SEG_CROP[1]/4)
    z = zs[choose] + random.randint(-SEG_CROP[2]/4,SEG_CROP[2]/4)
    pos = [x,y,z]
    label_crop = crop(label,pos)
    image_crop  = crop(image,pos)
    return image_crop,label_crop

def plot_patch(image,label):
    c = np.array(image.shape)/2
    plt.subplot(211)
    plt.imshow(np.squeeze(image[:,:,c[2]]),cmap=plt.cm.gray)
    plt.subplot(212)
    plt.imshow(np.squeeze(label[:,:,c[2]]),cmap=plt.cm.gray)
    plt.show()

def plot3(image):
    c = np.array(image.shape)/2
    plt.subplot(311)
    plt.imshow(np.squeeze(image[c[0],:,:]),cmap=plt.cm.gray)
    plt.subplot(312)
    plt.imshow(np.squeeze(image[:,c[1],:]),cmap=plt.cm.gray)
    plt.subplot(313)
    plt.imshow(np.squeeze(image[:,:,c[2]]),cmap=plt.cm.gray)
    plt.show()



@threadsafe_generator
def generate_lidc(data,labels,neg_fraction=0.5,out=3):
    total = 1.
    neg = 0.
    PLOT = False
    while True:
        for i,image in enumerate(data):
            label = labels[i]

            empty = False
	    if neg/total > neg_fraction:
		if np.sum(label) > 0:
		# get nodule sample
		    image_crop,label_crop = find_nodule(image,label)
		else:
 		    continue # look for another positive sample
            else: # get negative sample
                count = 0
                while not empty and count<5:
                    margin = 0
                    image = data[i]
                    x = random.randint(margin,image.shape[0] - margin)
                    y = random.randint(margin,image.shape[1] - margin)
                    z = random.randint(margin,image.shape[2] - margin)
                    pos = (x,y,z)
                    image_crop = crop(image,pos)
                    label_crop = crop(label,pos)
                    if np.sum(label_crop) > 0 :
                            count += 1
                    else: 
                        empty = True
                        neg += 1
                if not empty:
                    continue
            total += 1.	
            image_crop = image_crop.astype(np.float32)/MAX_PIXEL
            image_crop,label_crop = augment(image_crop,label_crop)
            if PLOT:
                plot_patch(image_crop,label_crop)
            image_crop = np.expand_dims(image_crop,axis=3)
            label_crop = np.expand_dims(label_crop,axis=3)

            if out>1:
                label_s = seg_downsample(label_crop,mode='maxpool')

                label_ss = seg_downsample(label_s,mode='maxpool')
                label_detect = np.amax(label_crop,keepdims=True)
                #print label_detect.shape
                yield image_crop,label_ss,label_s,label_crop,label_detect
            else:
                yield image_crop,label_crop
 
def generate_lidc_refine(data,labels,detections,out=3):
    pos_fraction = 0.5
    total = 1.
    pos = 0.
    pad_image = np.zeros(NEW_SHAPE,dtype=np.uint16)
    pad_label = np.zeros(NEW_SHAPE,dtype=np.uint8)
    while True:
        for i in range(len(data)):
            image = pad(data[i],pad_image)
            label = pad(labels[i],pad_label)
	    if pos/total < pos_fraction:
		if np.sum(label) > 0:
		# get nodule sample
		    image_crop,label_crop = find_nodule(image,label)
                    pos += 1
		else:
 		    continue # look for another positive sample
            else: # get previous detection
                if len(detections[i])>0:
                    ind = random.randint(0,len(detections[i])-1)
                    position = np.array(detections[i][ind])
                    position += np.random.randint(-4,4,size=position.shape)
                    image_crop = crop(image,position)
                    label_crop = crop(label,position)
                    if np.sum(label_crop) > 0 :
                            pos += 1
                else:
                    continue
            total += 1.	
            image_crop = image_crop.astype(np.float32)/MAX_PIXEL
            image_crop,label_crop = augment(image_crop,label_crop)
            image_crop = np.expand_dims(image_crop,axis=3)
            label_crop = np.expand_dims(label_crop,axis=3)
            if out>1:
                label_s = seg_downsample(label_crop,mode='maxpool')
                label_ss = seg_downsample(label_s,mode='maxpool')
                label_detect = np.amax(label_crop,keepdims=True)
                yield image_crop,label_ss,label_s,label_crop,label_detect
 
            else:
                yield image_crop,label_crop
 
@threadsafe_generator
def generate_lidc_batch(data,labels,batch_size=1,detections=[],neg_fraction=0.5,out=3):
    if len(detections)>0:
        seq=iter(generate_lidc_refine(data,labels,detections,out=out))
    else:
        seq=iter(generate_lidc(data,labels,neg_fraction,out=out))
    while True:
        inputs=[]
        if out>1:
           target1=[]
           target2=[]
        target3=[]
        detect=[]
        for i in range(batch_size):
            if out>1:
                x,y1,y2,y3,y4=seq.next()
                inputs.append(x)
                target1.append(y1)
                target2.append(y2)
                target3.append(y3)
                detect.append(y4)
            else:
                x,y3 =seq.next()
                inputs.append(x)
                target3.append(y3)
    
        inputs = np.asarray(inputs)
        if out>1:
            target1 = np.asarray(target1)
            target2 = np.asarray(target2)
        target3 = np.asarray(target3)
        detect = np.asarray(detect)
        if out>1:
            result = ({'inputs':inputs},{'l1':target1,
                'l2': target2,'l3':target3,'detect':detect})
        else:
            result = ({'inputs':inputs},{'l3':target3})
        yield result

@threadsafe_generator
def generate_dsb(data,labels,rand):
    while True:
        order = range(len(data))
        if rand:
            random.shuffle(order)
        for i in order:
            features = pad(data[i]).astype(np.float32)/MAX_PIXEL
            features = np.expand_dims(image,4)
            if rand:
                features = augment(features)
            yield (features,labels[i])

@threadsafe_generator
def generate_dsb_batch(data,labels,rand,batch_size=1):

    seq=iter(generate_dsb(data,labels,rand))
    while True:
        inputs=[]
        targets=[]
        for _ in range(batch_size):
            x,y=seq.next()
            inputs.append(x)
            targets.append(y)

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        result = ({'inputs':inputs},{'output':targets})
        yield result

def load_numpy_images(dataset='train'):
    labels = pd.read_csv(LABELS_FILE) 
    names = []
    if dataset == 'train':
        data = []
        for index,row in tqdm(labels.iterrows()):
            patient = row['id']
            label = row['cancer']
            image = np.load(PROCESSED_IMAGES_ROOT+patient+'.npy')
            image = crop_box(image)
            data.append(image.astype(np.uint16))
	    names.append(patient)
        cancer_labels = np.squeeze(labels.as_matrix(columns=['cancer']))
    else:
        data = []
        ids = [] # fill with ids for test data
        patients=os.listdir(ORIGINAL_IMAGES_ROOT)
        for patient in patients:
            if patient not in labels.id.values:
                image = np.load(PROCESSED_IMAGES_ROOT+patient+'.npy')
                image = crop_box(image)
                data.append(image)
                ids.append(patient)
		names.append(patient)
        cancer_labels = ids 
  
    max_shape = np.max(np.array([x.shape for x in data]),axis=0)
    print "Max shape",max_shape
       
    data = np.asarray(data)
    combined =  list(zip(data,cancer_labels,names))
    random.shuffle(combined)
    data[:],cancer_labels[:],names[:]=zip(*combined)
 
    return data,cancer_labels,names

def load_scan(path):
    #print "Path",path
    slices1 = [dicom.read_file(s) for s in glob.glob(path+'/*.dcm')]
    slices1.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    z0 = -9999
    dz = 0
    non_duplicate=[]
    for i,s in enumerate(slices1):
        z = float(s.ImagePositionPatient[2])
        if abs(z-z0) >= dz-0.01:
            non_duplicate.append(i)
            if (dz==0 and i>0):
                dz = z-z0
            z0 = z
    images = []
    slices = [slices1[i] for i in non_duplicate]
    for s in slices:
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        # Convert to Hounsfield units (HU)
        slice_image = s.pixel_array.astype(float)
        #print slope,intercept,np.min(slice_image)
        if slope != 1:
            slice_image *= slope
        slice_image += intercept
        #print s.ImagePositionPatient[2]
        images.append(slice_image)
    image = np.stack(images)
    slice_thickness = np.abs(float(slices[-1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))/float(len(slices)-1)
    #print slice_thickness
    for s1 in slices:
        s1.SliceThickness = slice_thickness
        
    return slices,image
	
def get_pixels_hu(image,scans):
   # Set outside-of-scan pixels to air (-1000 after converting to HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    #print np.min(image),np.median(image),np.max(image),intercept,slope
    out = np.where(image == -2000)

    # Convert to Hounsfield units (HU)
    #print intercept,slope,image.dtype
    if slope != 1:
        image = slope * image 
    image += int(intercept)
    image[out]= -1000

    #print np.min(image),np.median(image)
    if intercept == -2048 and np.median(image)<-1000:
        print "Anomalous intercept"
        image += -1024 -int(np.min(image))
        print "after",np.min(image),np.median(image)
    	
    return image

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = MAX_PIXEL*(image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>MAX_PIXEL] = MAX_PIXEL
    image[image<0] = 0.
    image = np.round(image).astype(np.uint16)	
    return image


def reshape(image,original_spacing,new_spacing= NEW_SPACING):
    # input z axis is along 0 dimension, output along 2'nd dimension
    image = np.transpose(image,[2,1,0])
    original_spacing = original_spacing[[2,1,0]]
    resize_factor=original_spacing/new_spacing
    expected_size = np.round(np.array(image.shape)*resize_factor)
    image = scipy.ndimage.interpolation.zoom(image,resize_factor)
    return image,new_spacing


def plot_image(image):
    plt.subplot(311)
    plt.imshow(np.squeeze(image[image.shape[0]/2,:,:]),cmap=plt.cm.gray)  
    plt.subplot(312)
    plt.imshow(np.squeeze(image[:,image.shape[1]/4,:]),cmap=plt.cm.gray)  
    plt.subplot(313)
    plt.imshow(np.squeeze(image[:,:,image.shape[2]/2]),cmap=plt.cm.gray)  
    plt.show()

def process_image(path,info=False):
    scan,image = load_scan(path)
    patient = path.split('/')[-1]
    original_spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    orientation = [float(x) for x in scan[0].ImageOrientationPatient]
    original_spacing = np.array(list(original_spacing))
    if not info and orientation[1]<0:
        image = image[:,::-1,::-1]
    #image = image.astype(np.float32) # temporary
    #image = get_pixels_hu(image,scan)
    image = normalize(image)
    
    image,new_spacing = reshape(image,original_spacing)

    pre_shape = image.shape
    if not info:
        image,_,_ = get_lung_region(image)
        np.save(PROCESSED_IMAGES_ROOT+patient+'.npy',image)
    PLOT = False
     
    if PLOT and (image.shape[0] == image.shape[1] or image.shape[0]==image.shape[2]) and image.shape[0]>160:
        #print path,"reshaped",pre_shape,"final",image.shape
        plot_image(image)
        # axis 2 is  head/legs
    if info:
        return image,new_spacing,orientation

@threadsafe_generator
def generate_detect(data,labels,rand):
    while True:
        order = range(len(data))
        if rand:
            random.shuffle(order)
        for i in order:
            #image =np.squeeze(image).astype(np.float32)/MAX_PIXEL
            image = data[i]
            label = labels[i]
            #if rand:
            #    x = random.randint(0,image.shape[0]-det_crop[0])
            #    y = random.randint(0,image.shape[1]-det_crop[1])
            #    z = random.randint(0,image.shape[2]-det_crop[2])
            #else:
            #    x = (image.shape[0]-det_crop[0])/2
            #    y = (image.shape[1]-det_crop[1])/2
            #    z = (image.shape[2]-det_crop[2])/2
		
            image_crop = np.squeeze(image)
            #print image.shape
            #image_crop = seg_crop(image,(x,y,z),det_crop)
            if rand:
                #pass
                image_crop = augment1(image_crop)
            #print image_crop.shape,image_crop.dtype,np.mean(image_crop)
            #if rand:
                #for k in range(image.shape[3]):
                    #image_crop[:,:,:,k]=augment1(np.squeeze(image_crop[:,:,:,k]))
                    #plot3(np.squeeze(image_crop[:,:,:,k]))
            #image_crop = np.expand_dims(image_crop,axis=3)
            yield (image_crop,label)

@threadsafe_generator
def generate_detect_batch(data,labels,rand,batch_size=1):

    seq=iter(generate_detect(data,labels,rand))
    while True:
        inputs=[]
        targets=[]
        for _ in range(batch_size):
            x,y=seq.next()
            inputs.append(x)
            targets.append(y)

        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        #targets = np.expand_dims(targets,axis=1)
        #print targets.shape
        #print targets
        result = ({'inputs':inputs},{'output':targets})
        yield result

def load_numpy_detections(dataset='train'):
    labels = pd.read_csv(LABELS_FILE)
    if dataset == 'train':
        data = np.zeros((1400,5,5,4,151),dtype=np.float32)
        
        for index,row in tqdm(labels.iterrows()):
            patient = row['id']
            label = row['cancer']
            detection = np.load(DETECTIONS_DSB_ROOT+dataset+'/features_'+patient+'.npy')
            detection = detection.astype(np.float32)
            data[index,:,:,:,:]=detection
        cancer_labels = np.squeeze(labels.as_matrix(columns=['cancer']))
        data = data[:(index+1),...]

        combined =  list(zip(data.tolist(),cancer_labels.tolist()))
        random.shuffle(combined)
        data[...],cancer_labels[...]=zip(*combined)
        data = np.array(data)
        lables = np.array(labels)

    else:
        data = []
        ids = [] # fill with ids for test data
        patients=os.listdir(ORIGINAL_IMAGES_ROOT)
        for patient in patients:
            if patient not in labels.id.values:
                detection = np.load(DETECTIONS_DSB_ROOT+dataset+'/features_'+patient+'.npy')
                #detection = np.transpose(np.squeeze(detection),(1,2,3,0))
                detection = detection.astype(np.float32)
                data.append(detection)
                ids.append(patient)
        cancer_labels = ids 
        data = np.asarray(data)
     
    return data,cancer_labels

if __name__ == "__main__":
    print('Usage: %s ORIGINAL_IMAGES_ROOT/ PROCESSED_IMAGES_ROOT/ LABELS_FILE' % sys.argv[0])
    if len(sys.argv) == 4:
        ORIGINAL_IMAGES_ROOT = sys.argv[1]
        PROCESSED_IMAGES_ROOT = sys.argv[2]
        LABELS_FILE = sys.argv[3]
        if not os.path.exists(PROCESSED_IMAGES_ROOT):
            os.makedirs(PROCESSED_IMAGES_ROOT)

    patients=os.listdir(ORIGINAL_IMAGES_ROOT)
    patients.sort()
    #patients = ['fb7dfb6aae597d97c2da24179df0fe56','d833b4c1530183c1b3eae02e9a2cd735','5ab68460985d9ed8e952475b402ddd62','169b5bde441e8aa3df2766e2e02cda08','043ed6cb6054cc13804a3dca342fa4d0']
    Parallel(n_jobs=-1,verbose=1)(delayed(process_image)(ORIGINAL_IMAGES_ROOT+patient) for patient in patients)

