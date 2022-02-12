#!/bin/bash
# remember to add --save when you want to save the experiment log
# ./run_batch_jf17k.sh 0 jf17k statements StarE_grad stare_stats_baseline

GPU=$1

dataset=$2
datatype=$3
encoder=$4
model=$5
batch=1024
data_aug="None"
wandb=$6 # True or False: use wandb to keep training logs or not
seed=1996 # random seed
qua_prob=0.0 # probability of masking qualifier entities. Used for statement completion task 

if [ ${model} == "stare_transformer" ]
then
    lr="0.001"
else
    lr="0.0001"
fi

if [ ${seed} == "all" ]
then
    SEED=(1996 42 62 2021 15213)
else
    SEED=($seed)
fi

for seed in "${SEED[@]}"
do
    option="DEVICE cuda DATASET ${dataset} EVAL_EVERY 10
            BATCH_SIZE ${batch} LEARNING_RATE ${lr} WANDB ${wandb}
            SUBTYPE ${datatype} EPOCHS 401 GCN_ENCODER ${encoder}
            MODEL_NAME ${model} SEED ${seed}
            GCN_SEP_ENT_EMBEDDING False
            GCN_WEIGHT_TRANS True GCN_MASK_EDGE False
            USE_TEST True CLEANED_DATASET False
            DATA_AUG ${data_aug} GCN_POSITIONAL True QUAL_PROB ${qua_prob}"
    cmd="CUDA_VISIBLE_DEVICES=${GPU} python run.py ${option}"
    echo $cmd
    eval $cmd
done
