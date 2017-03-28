import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import pylidc as pl
import warnings
import pickle
from tqdm import tqdm
import glob
import random
import threading

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed


NEW_SPACING= (1.4,1.4,2.)
PROCESSED_DIR = '/home/ronens1/lidc/processed/'
PIXEL_RANGE = 65535
CROP_SHAPE = (32,32,32)
random.seed(1)
NODULE_THRESHOLD = 0.5

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


def plot(vol,x,y,z):
    corner1 = np.array([x,y,z])-np.array(CROP_SHAPE)/2
    corner2 = corner1+np.array(CROP_SHAPE)
    plt.subplot(311)
    plt.imshow(vol[x,corner1[1]:corner2[1],corner1[2]:corner2[2]],cmap=plt.cm.gray)
    plt.subplot(312)
    plt.imshow(vol[corner1[0]:corner2[0],y,corner1[2]:corner2[2]],cmap=plt.cm.gray)
    plt.subplot(313)
    plt.imshow(vol[corner1[0]:corner2[0],corner1[1]:corner2[1],z],cmap=plt.cm.gray)
    plt.show()	
   

def process_scan(scan):
    uid = scan.series_instance_uid
    volume,spacing,orientation,z0 = scan.to_volume()
    volume = volume.transpose([1,0,2])    
    if orientation[0]<0:
        volume=volume[::-1,::-1,:]
    resize_factor = np.array(spacing)/np.array(NEW_SPACING)
    resampled = scipy.ndimage.interpolation.zoom(volume,resize_factor) 
    resampled = normalize(resampled)
    shape = resampled.shape
    clusters = scan.annotations_with_matching_overlap()
    clusters_data=[]
    for cluster in clusters:
        cluster_group=[]
        for ann in cluster:
            diameter = ann.estimate_diameter()
            features = ann.feature_vals()
            c = ann.centroid()
            c[:2]=c[:2]*np.array(spacing[:2])
            c[2] = c[2]-z0
            c = c/np.array(NEW_SPACING)
            b = ann.bbox()
            b[:2,:] = b[:2,:]*np.expand_dims(np.array(spacing[:2]),axis=1)
            b[2,:] = b[2,:]-z0
            b = b / np.expand_dims(np.array(NEW_SPACING),axis=1)
            if orientation[0]<0:
                c[:2] = np.array(resampled.shape)[:2] - c[:2]
                b[:2,:] = np.expand_dims(np.array(resampled.shape)[:2],axis=1)-b[:2,:]
            #plot(resampled,int(c[0]),int(c[1]),int(c[2]))
            annotation= {'diameter': diameter,'features':features, 'centroid':c,'bbox':b}
            cluster_group.append(annotation)
            if c[2]<0 or b[2,0]<0 or b[2,1]<0:
                print "Error",uid,orientation,c,b,ann.centroid(),ann.bbox()
        clusters_data.append(cluster_group)

    np.save(PROCESSED_DIR+uid+'.npy',resampled)
    with open(PROCESSED_DIR+uid+'annotation.txt', 'w') as outfile:
            pickle.dump(clusters_data, outfile)
def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = PIXEL_RANGE*(image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>PIXEL_RANGE] = PIXEL_RANGE
    image[image<0] = 0.
    image = np.round(image).astype(np.uint16)
    return image

def load_lidc():
    filenames = glob.glob(PROCESSED_DIR+'*.npy')
    data = []
    annotations = []
    for name in tqdm(filenames):
        data.append(np.load(name))
        annotation_file_name = '.'.join(name.split('.')[:-1])+'annotation.txt'
	with open(annotation_file_name,'r') as pickle_file:
        	annotations.append(pickle.load(pickle_file))
    perm = range(len(annotations))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    annotations= [annotations[i] for i in perm]
    data=np.asarray(data)
    return data,annotations


def soft_focus(pos,centroid):
    Focus_radius = 30 # mm
    dist = np.linalg.norm( (np.array(pos)-np.array(centroid))*np.array(NEW_SPACING))/Focus_radius
    return max(1-dist,0)


def shift_radius(shift):
 r = 100
 while r > shift:
     v = np.random.uniform(-shift,high=shift,size=(3,))
     r = np.linalg.norm(v)
 vox_shift = (v/np.array(NEW_SPACING)).astype(int)
 return vox_shift   

def find_nodule(annotation,min_agreement):
    good_clusters = [cluster for cluster in annotation if len(cluster)>=min_agreement]
    marks = [mark for cluster in good_clusters for mark in cluster]
    mark = marks[random.randint(0,len(marks)-1)]
    centroid = np.array(mark['centroid']).astype(int) 

    shift = 12.0 # mm , shold be within soft noudle detection threshold
    pos = centroid + shift_radius(shift)
    #print "diameter",mark['diameter']
    """
    feature_names = \
       ('subtlety',
	'internalStructure',
	'calcification',
	'sphericity',
	'margin',
	'lobulation',
	'spiculation',
	'texture',
	'malignancy')
    """
    soft = soft_focus(pos,centroid)
    if (soft < NODULE_THRESHOLD) :
        print 'Error: nodule shifted too much'
    malig = mark['features'][8]
    diameter = mark['diameter']
    return pos,np.array([soft,malig/5.0,diameter])	

def plot_patch(image):
    c = np.array(image.shape)/2
    plt.subplot(311)
    plt.imshow(np.squeeze(image[c[0],:,:]),cmap=plt.cm.gray)
    plt.subplot(312)
    plt.imshow(np.squeeze(image[:,c[1],:]),cmap=plt.cm.gray)
    plt.subplot(313)
    plt.imshow(np.squeeze(image[:,:,c[2]]),cmap=plt.cm.gray)
    plt.show()

def crop(image,position):
    corner1 = np.array(position)-np.array(CROP_SHAPE)/2
    corner1 = np.maximum(corner1, np.array([0,0,0]))
    corner2 = corner1+np.array(CROP_SHAPE)
    corner2 = np.minimum(corner2,np.array(image.shape))
    corner1 = corner2-np.array(CROP_SHAPE)
    patch = image[corner1[0]:corner2[0],corner1[1]:corner2[1],corner1[2]:corner2[2]]
    return patch

def bbox_in_patch(bbox,pos):
    corner1 = np.array(pos) - np.array(CROP_SHAPE)/2
    corner2 = corner1 + np.array(CROP_SHAPE)/2
    if np.all(bbox[:,0] > corner1) and np.all(bbox[:,1] < corner2):
        nodule = True
    else:
        nodule = False
    return nodule

def check_centroid(centroid,pos):
    check = False
    diff = np.abs(np.array(centroid)-np.array(pos))
    if np.all(diff < np.array(CROP_SHAPE)/4):
        #print "check_centroid",diff, CROP_SHAPE
        check = True
    return check
def find_label(pos,annotation,min_agreement):
    nodule = 0
    malig = 0
    biggest_diameter = 0
    c  = 0
    for cluster in annotation:
        if len(cluster) >= min_agreement:
        # choose randomly one mark from each cluster
            mark = cluster[random.randint(0,len(cluster)-1)]
            #bbox = mark['bbox']
            #if bbox_in_patch(bbox,pos):
            centroid = mark['centroid']
            soft = soft_focus(centroid,pos)
            if soft  > NODULE_THRESHOLD:
                diameter = mark['diameter']
                if diameter > biggest_diameter:
                    biggest_diameter = diameter
                    malig = mark['features'][8]
                    nodule = soft
                    c = np.array(centroid).astype(int)
    #if nodule:
        #print "find_label",biggest_diameter,pos,c

    return np.array([nodule,malig/5.0,biggest_diameter]),c

def augment(patch):
    if random.random() < 0.5:
        patch = patch[::-1,:,:]
    if random.random() < 0.5:
        patch = patch[:,::-1,:]
    if random.random() < 0.5:
        patch = patch[:,:,::-1]
    perm = [0,1]
    random.shuffle(perm)
    patch = np.transpose(patch,perm+[2])
    return patch

def check_agreement(annotation,minimum):
    n = 0
    if len(annotation)>0:
	n = [ len(x) for x in annotation]
	ok =  np.max(n) >= minimum
    else:
        ok = False
    #print "check agreement",minimum,np.max(n),ok
    return ok	

@threadsafe_generator
def generate_lidc(data,annotations):
    neg_fraction = 0.5
    total = 1.
    neg = 0.
    min_agreement = 3
    PLOT = False
    skip = 0
    while True:
        for i in range(len(annotations)):
            random_sample = False
            annotation = annotations[i]
            if neg/total > neg_fraction: 
	        if check_agreement(annotation,min_agreement):				
                    # get positive sample
                    pos,label = find_nodule(annotation,min_agreement)
                    image = data[i]
                    patch = crop(image,pos)
                else:
                    skip += 1
                    #print total,skip
                    continue # continue to find another image with some nodule
            else:
                # get random sample
                random_sample = True
                margin = 30
                image = data[i]
                x = random.randint(margin,image.shape[0] - margin)
                y = random.randint(margin,image.shape[1] - margin)
                z = random.randint(margin,image.shape[2] - margin)
                pos = (x,y,z)
                label,centroid = find_label(pos,annotation,min_agreement)
                #if label[0]==1:
                #    pos = centroid
                patch = crop(image,pos)
                if label[0] == 0:
                    neg += 1
            total += 1
            # random augmentation

            patch = patch.astype(np.float32)/PIXEL_RANGE
            patch = augment(patch)
            if PLOT:
                print i,label, patch.shape,np.max(patch)
                plot_patch(patch)


            patch = np.expand_dims(patch,axis=4)
            yield patch,label

@threadsafe_generator
def generate_lidc_batch(data,annotations,batch_size=1):

    seq=iter(generate_lidc(data,annotations))
    while True:
        inputs=[]
        targets1=[]
	targets2=[]	
        targets3=[]
        for _ in range(batch_size):
            x, y = seq.next()
            inputs.append(x)
            targets1.append(y[0])
	    targets2.append(y[1])
            targets3.append(y[2])    
        inputs = np.asarray(inputs)
        targets1 = np.asarray(targets1)
        targets2 = np.asarray(targets2)
        targets3 = np.asarray(targets3)/10. # mm -> cm

        result = ({'inputs':inputs},{'nodule':targets1,'malig':targets2,'diameter': targets3})

        yield result



if __name__ == "__main__":
    qu = pl.query(pl.Scan)
    print qu.count(),'scans'
    for scan in tqdm(qu):
        process_scan(scan)
