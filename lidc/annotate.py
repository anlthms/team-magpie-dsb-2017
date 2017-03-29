import numpy as np
import pylidc as pl
import warnings
from tqdm import tqdm
import pandas as pd
import csv
import dicom
import glob
import os
import sys
import SimpleITK as sitk

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed

LIDC_ROOT = '/mnt/storage/forShai/LIDC/'
PROCESSED_DIR = '/home/ronens1/lidc/processed/'
min_agreement = 3

def get_annotations(scan):
    uid = scan.series_instance_uid
    clusters = scan.annotations_with_matching_overlap(tol=0.8)
    clusters_data=[]
    for cluster in clusters:
        if len(cluster)<min_agreement:
            continue
        diameter = np.zeros(1)
        features = np.zeros(9)
        centroid = np.zeros(3)
        for ann in cluster:
            diameter += ann.estimate_diameter()
            features += np.array(ann.feature_vals())
            centroid += np.array(ann.centroid())
        n = float(len(cluster))
        diameter /= n
        features /= n
        centroid /= n
        clusters_data.append({'uid':uid,'diameter':diameter,'features':features,'centroid':centroid})
    return clusters_data


def transform_centroid(row):
    uid = str(row[0])
    centroid = row[1:4]
    files = glob.glob(LIDC_ROOT + '*/*/' + uid + '/*.dcm')
    slices = [dicom.read_file(s) for s in files]
    positions = np.unique([int(x.ImagePositionPatient[2]) for x in slices])
    slice_thickness = np.abs(positions[1] - positions[0])
    z0 = positions[0]
    centroid[2] = (centroid[2]-z0)
    pixel_spacing = slices[0].PixelSpacing
    centroid[0] *= float(pixel_spacing[0])
    centroid[1] *= float(pixel_spacing[1])
    return centroid

if __name__ == "__main__":
    print('Usage: %s LIDC_ROOT/ PROCESSED_DIR/' % sys.argv[0])
    if len(sys.argv) == 3:
        LIDC_ROOT = sys.argv[1]
        PROCESSED_DIR = sys.argv[2]
    CSV_DIR = os.path.dirname(PROCESSED_DIR[:-1])+'/csv/'
    qu = pl.query(pl.Scan)
    rows = []
    features_dict = ['subtlety','internalStructure','calcification','sphericity',
            'margin','lobulation','spiculation','texutre','malignancy']
    uids =[]
    for scan in tqdm(qu):
        uids.append(scan.series_instance_uid)
        clusters = get_annotations(scan)
        for annotation in clusters:
            row = [annotation['uid']]+annotation['centroid'].tolist()+annotation['diameter'].tolist()+annotation['features'].tolist()
            rows.append(row)
    df = pd.DataFrame(uids)
    df.to_csv(CSV_DIR + 'list.csv',index=False,header=False)
    transformed_centroids  = Parallel(n_jobs=12,verbose=1)(delayed(transform_centroid)(row[0:4]) for row in rows)
    for i,row in enumerate(rows):
        row[1:4] = transformed_centroids[i]
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR + 'annotations.csv',index=False,header=False)

