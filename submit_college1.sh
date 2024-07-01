#!/bin/bash

#SBATCH -J "dphm_dice_focal"
#SBATCH -p dlq
#SBATCH -o /home/cs14274101/dj/log/dphm_dice_focal.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 150:00:00
#SBATCH -x compute[02-03]

/home/cs14274101/miniconda3/envs/py11/bin/python -u /home/cs14274101/dj/segmentation/train_vanilla.py --cfg /home/cs14274101/dj/segmentation/cfgs/training/college1_Dphm.toml
