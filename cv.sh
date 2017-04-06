#!/bin/bash
for i in `seq 1 9`
do
    echo fold $i
    PYTHONPATH=. python train_detect.py $DATA_ROOT/dsb/stage1/ $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1_labels.csv $i
done
echo fold 0
PYTHONPATH=. python train_detect.py $DATA_ROOT/dsb/stage1/ $DATA_ROOT/dsb/processed/ $DATA_ROOT/dsb/stage1_labels.csv 0 
