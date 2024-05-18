#!/bin/bash

###
 # @Author: Hadlay Zhang
 # @Date: 2024-05-12 00:33:48
 # @LastEditors: Hadlay Zhang
 # @LastEditTime: 2024-05-18 22:46:21
 # @FilePath: /root/MedicalVQA-RAD/run.sh
 # @Description: Running Shell Script for training and testing
### 

###############
# 1. If only running a single train or test
###############
if [ "$1" == "train" ]; then
    python3 main.py --use_RAD --RAD_dir data_RAD/ --text BioBERT --text_path /root/autodl-tmp/BioBERT-v1.1/ --image SwinTV2B --output /root/autodl-tmp/RAD/test/saved_models/T1/
elif [ "$1" == "test" ]; then
    python3 test.py --use_RAD --RAD_dir data_RAD/ --text BioBERT --text_path /root/autodl-tmp/BioBERT-v1.1/ --image SwinTV2B --input /root/autodl-tmp/RAD/test/saved_models/T1/ --epoch 19 --output /root/autodl-tmp/RAD/test/results/T1/
else
    echo "Usage: $0 [train|test]"
    exit 1
fi

###############
# 2. If running for several iterations
###############
# paths
# image="ConvNeXt" # Image Encoder
# text="BioBERT" # Text Encoder
# text_path="/root/autodl-tmp/BioBERT-v1.1/"
# counter=5
# # <modify local paths>
# prefix="/root/autodl-tmp/RAD/"
# models_dir="saved_models/"
# result_dir="results/"
# model_name="ConvNeXt-BioBERT-BAN/"
# # random seeds
# seeds=(1204 88 308 12345 9999)

# for i in $(seq 1 $counter); do
#     sub="T$i/"
#     model_path="${prefix}${models_dir}${model_name}${sub}"
#     results_path="${prefix}${result_dir}${model_name}${sub}"
#     seed=${seeds[$i-1]}

#     # Training
#     echo "Training iteration $i, current seed: $seed, Models will be saved in: $model_path"
#     python3 main.py --use_RAD --RAD_dir data_RAD/ --text $text --text_path $text_path --image $image --seed $seed --output $model_path

#     # Testing
#     echo "Testing iteration $i, Results will be saved in: $results_path"
#     python3 test.py --use_RAD --RAD_dir data_RAD/ --text $text --text_path $text_path --image $image --epoch 19 --input $model_path --output $results_path
# done
