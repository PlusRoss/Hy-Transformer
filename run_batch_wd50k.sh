#!/bin/bash
# remember to add --save when you want to save the experiment log
# ./run_batch_wd50k.sh 0 wd50k_100 statements StarE_grad stare_transformer
# ./run_batch_wd50k.sh 0 wd50k_100 statements StarE_grad stare_stats_baseline
# ./run_batch_wd50k.sh 4 wd50k_100 statements StarE stare_stats_baseline_well None True
# stare_transformer

GPU=$1

DATASET=$2
datatype=$3
encoder=$4
model=$5
data_aug="None"
wandb=$6 # True or False: use wandb to keep training logs or not
seed=1996 # random seed
qua_prob=0.0 # probability of masking qualifier entities. Used for statement completion task 


if [ ${DATASET} == "all" ]
then
    DATASETS=("wd50k" "wd50k_33" "wd50k_66" "wd50k_100")
else
    DATASETS=(${DATASET})
fi

if [ ${seed} == "all" ]
then
    SEED=(1996 42 62 2021 15213)
else
    SEED=($seed)
fi

for dataset in "${DATASETS[@]}"
do
    for seed in "${SEED[@]}"
    do
        lr="0.0001"
        batch=512 # default: 512
        option="DEVICE cuda DATASET ${dataset} EVAL_EVERY 10
                BATCH_SIZE ${batch} LEARNING_RATE ${lr} WANDB ${wandb}
                SUBTYPE ${datatype} EPOCHS 401 GCN_ENCODER ${encoder}
                MODEL_NAME ${model} SAVE False SEED ${seed}
                GCN_SEP_ENT_EMBEDDING False
                GCN_WEIGHT_TRANS True GCN_MASK_EDGE False
                USE_TEST True 
                GCN_FEAT_DROP 0.3 GCN_T_LAYERS 2 WEIGHT_DECAY 0
                DATA_AUG ${data_aug} GCN_POSITIONAL True QUAL_PROB ${qua_prob}"
        cmd="CUDA_VISIBLE_DEVICES=${GPU} python run.py ${option}"
        echo $cmd
        eval $cmd
    done
done
