#!/bin/bash

###
 # @Author: Hadlay Zhang
 # @Date: 2024-05-12 00:33:48
 # @LastEditors: Hadlay Zhang
 # @LastEditTime: 2024-05-31 00:38:01
 # @FilePath: /root/MedicalVQA-RAD/run.sh
 # @Description: Running Shell Script for training and testing
### 

image="ConvNeXt" # Image Encoder, [ViTL16, ConvNeXt, SwinTV2B]
text="BioBERT" # Text Encoder, [BioBERT, BERT]
# <pretrained model path>
text_path="/root/autodl-tmp/BioBERT-v1.1/" # for BioBERT
# text_path="/root/autodl-tmp/BERT/" # for BERT 
attention="BAN"
epochs=20
last_epoch=$(($epochs - 1))
# <modify local paths>
prefix="/root/autodl-tmp/RAD/"
models_dir="saved_models/"
result_dir="results/"
model_name="${image}-${text}-${attention}-${epochs}epochs/"

###############
# 1. If only running a single train or test
###############
model_path="${prefix}${models_dir}${model_name}T1/"
results_path="${prefix}${result_dir}${model_name}T1/"
if [ "$1" == "train" ]; then
    python3 main.py --use_RAD --RAD_dir data_RAD/ --epochs $epochs --text $text --text_path $text_path --image $image --att $attention --output $model_path
elif [ "$1" == "test" ]; then
    python3 test.py --use_RAD --RAD_dir data_RAD/ --epoch $last_epoch --text $text --text_path $text_path --image $image --att $attention --input $model_path --output $results_path
else
    echo "Usage: $0 [train|test]"
    exit 1
fi

# ###############
# # 2. If running for several iterations using fixed seeds
# ###############

# counter=10
# start=1
# # random seeds
# # seeds=(1204 88 308 12345 9999) # five iters
# seeds=(670487 116739 262949 777572 625049 168726 971963 196842 967067 619993) # ten iters

# for i in $(seq $start $counter); do
#     sub="T$i/"
#     model_path="${prefix}${models_dir}${model_name}${sub}"
#     results_path="${prefix}${result_dir}${model_name}${sub}"
#     seed=${seeds[$i-1]}

#     # Training
#     echo "Training iteration $i, current seed: $seed, Models will be saved in: $model_path"
#     python3 main.py --use_RAD --RAD_dir data_RAD/ --epochs $epochs --text $text --text_path $text_path --image $image --att $attention --seed $seed --output $model_path

#     # Testing
#     echo "Testing iteration $i, Results will be saved in: $results_path"
#     python3 test.py --use_RAD --RAD_dir data_RAD/ --epoch $last_epoch --text $text --text_path $text_path --image $image --att $attention --input $model_path --output $results_path
# done
