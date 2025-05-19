#!/bin/bash
set -x

# 正确的变量赋值方式
attack="$1"
gpu="$2"

# 按顺序执行 Python 脚本
python main.py --attack "$attack" --model resnet18 --GPU_ID "$gpu"
python main.py --attack "$attack" --model resnet18 --GPU_ID "$gpu" --eval
python main.py --attack "$attack" --model resnet101 --GPU_ID "$gpu"
python main.py --attack "$attack" --model resnet101 --GPU_ID "$gpu" --eval
python main.py --attack "$attack" --model resnext50_32x4d --GPU_ID "$gpu"
python main.py --attack "$attack" --model resnext50_32x4d --GPU_ID "$gpu" --eval
python main.py --attack "$attack" --model densenet121 --GPU_ID "$gpu"
python main.py --attack "$attack" --model densenet121 --GPU_ID "$gpu" --eval
