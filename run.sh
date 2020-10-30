#!/bin/bash

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=example

### 指定该作业需要多少个节点
#SBATCH --nodes=1

### 指定该作业需要多少个CPU
#SBATCH --ntasks=4

### 指定该作业在哪个队列上执行
### 目前可用的队列有 cpu/fat/titan/tesla
#SBATCH --partition=cpu

### 激活 Anaconda 环境
source activate python3

### 执行你的作业
python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset coco
