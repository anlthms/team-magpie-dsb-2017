#!/bin/bash -e

set -x
# The data is assumed to be inside $DATA_ROOT/lidc and $DATA_ROOT/dsb

DATA_ROOT=/usr/local/data

# convert LIDC
if [ ! -d $DATA_ROOT/lidc/processed/ ]
then
    PYTHONPATH=. python lidc/annotate.py $DATA_ROOT/lidc/DOI/ $DATA_ROOT/lidc/processed/
    PYTHONPATH=. python lidc/mypreprocess.py $DATA_ROOT/lidc/DOI/ $DATA_ROOT/lidc/processed/
fi

if [ ! -d ../data ]
then
    ln -s $DATA_ROOT/lidc/ ../data
fi

# convert DSB
if [ ! -d $DATA_ROOT/dsb/processed/ ]
then
    python preprocess1.py $DATA_ROOT/dsb/stage1/ $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1_labels.csv
fi

# convert DSB 2nd stage
if [ ! -d $DATA_ROOT/dsb/processed2/ ]
then
    python preprocess1.py $DATA_ROOT/dsb/stage2/ $DATA_ROOT/dsb/processed2/ dummy
fi

# train on LIDC. 0 is for no REFINE
python seg_lidc.py 0 ../data/processed/

# detect on LIDC
python detect.py LIDC 0 ../data/processed/ dummy dummy dummy dummy

# refine on LIDC. 1 is for REFINE
python seg_lidc.py 1 ../data/processed/

#tail -198 $DATA_ROOT/dsb/stage1_solution.csv | cut -d',' -f1,2 >> $DATA_ROOT/dsb/stage1_labels.csv
# detect on DSB
python detect.py DSB 1 $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1/ $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1_labels.csv train
python detect.py DSB 1 dummy $DATA_ROOT/dsb/stage2/ $DATA_ROOT/dsb/processed2/ $DATA_ROOT/dsb/stage1_labels.csv test

# train/validate/test on DSB
PYTHONPATH=. python train_detect.py $DATA_ROOT/dsb/stage2/ $DATA_ROOT/dsb/processed2/ $DATA_ROOT/dsb/stage1_labels.csv

