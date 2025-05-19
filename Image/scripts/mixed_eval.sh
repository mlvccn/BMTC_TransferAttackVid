#!/bin/bash
set -x

# 正确的变量赋值方式
attack="$1"
gpu="$2"
mixer_data_path=""

# 按顺序执行 Python 脚本
python main.py --attack "$attack" --model resnet18 --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path"
python main.py --attack "$attack" --model resnet18 --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path" --eval
python main.py --attack "$attack" --model resnet101 --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path"
python main.py --attack "$attack" --model resnet101 --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path" --eval
python main.py --attack "$attack" --model resnext50_32x4d --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path"
python main.py --attack "$attack" --model resnext50_32x4d --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path" --eval
python main.py --attack "$attack" --model densenet121 --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path"
python main.py --attack "$attack" --model densenet121 --GPU_ID "$gpu" --mixer_data_dir "$mixer_data_path" --eval
