#!/bin/bash -e

DATA_ROOT=/usr/local/data

# convert LIDC
if [ ! -d $DATA_ROOT/lidc/processed/ ]
then
    PYTHONPATH=. python lidc/annotate.py $DATA_ROOT/lidc/DOI/ $DATA_ROOT/lidc/processed/
    PYTHONPATH=. python lidc/mypreprocess.py $DATA_ROOT/lidc/DOI/ $DATA_ROOT/lidc/processed/
fi

ln -s $DATA_ROOT/lidc/ ../data

# convert DSB
if [ ! -d $DATA_ROOT/dsb/processed/ ]
then
    python preprocess1.py $DATA_ROOT/dsb/stage1/ $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1_labels.csv
fi

# train on LIDC. 0 is for no REFINE
python seg_lidc.py 0 ../data/processed/

# detect on LIDC
python detect.py LIDC 0 ../data/processed/

# refine on LIDC. 1 is for REFINE
python seg_lidc.py 1 ../data/processed/

# detect on DSB
python detect.py DSB 1 $DATA_ROOT/dsb/processed/

# train/validate/test on DSB
python train_detect.py $DATA_ROOT/dsb/stage1/ $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1_labels.csv
