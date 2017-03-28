DSB code description
===========================

lidc sub directory - run mypreprocess.py there to preprocess lidc database (obtainable from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI. Need the 124GB dicom images and the xml annotation file). Lidc was basis for luna16 but luna16 did some normalization and selecting the best subset and the original dataset format is messy. I'm using a third party free package to extract lidc annotations from xml files. You may want instead to adapt this stage to luna16. I started with luna16 actually but didn't keep track of my older code. LIDC should enable better training (it has labels for malignancy rate and more) but I did not see better results. 


This directory (src1) - 


Pipeline is to run the following:
================================
mypreprocess.py in the lidc directory (can adapt to luna16) to preprocess LIDC dataset.
preprocess1.py, to preprocess DSB dataset.
seg_lidc.py, with hard coded flag REFINE=False
detect.py with hard coded flag DataSet='LIDC'
optional: run seg_lidc again with hard coded flag REFINE=True, will use the detections from detect.py to refine the segmentation again.
detect.py - with hard coded flag DataSet='DSB', will produce dense features and save to disk.
train_detect.py - will train on the dense features from detect.py, using DSB labels. Will create submission file for the DSB test data.


File descriptions
=======================

(not described are some old files in the directory that were not used for recent predictions. 
Not sure this list is complete. )

preprocess1.py - preprocess DSB images, using identical pipline to that used to preprocess LIDC or luna16.

Also includes several important utility functions including Keras data generators (for online batch augmentation). 

seg_lidc.py - train the segmentation network on pre-processed LIDC images (or you can adpat it to luna16). There is a REFINE hard coded flag. At first pass it is False. After training once and running detect.py (described below), you can change REFINE to True and run again. It will then use the first detections on the full image (with many flase positives) as hard exmaples to retrain a second time.


detect.py - segment full images using pre-trained segmentation net, then save the detected regions ('detection_...npy') as well as intermediate layer features ('features...npy'). Print some accuracy statistics. There is hard coded flag DataSet='LIDC' or 'DSB'. use LIDC to detect on LIDC (you can adapt this to luna16), then the detections may be used to retrain the segmentation net (hard example mining, since at first pass many are false positives). Use 'DSB' to extract and save the dense features used by train_detect.py for the final classification.


train_detect.py -train/test on DSB features which were previously extracted and saved to disk using a segmentation network train on LIDC (or luna16) dataset.

An idea to imporve is more data augmentation (since only ~1000 images). But it seems it will have to be offline as the pipeline first involve predicting the features using detect.py.

Other needed utility modules:
metrics.py
resnet3.py - 3D version of residual network blocks.
backend_updated.py - keras custom udpate for 3D deconvolution block (for segmentation)
deconv3D.py - keras custom update for 3D deconvolution block (for segmentation), used with backend_updated.py


average.py - utility to average two predictions (for ensemble)

