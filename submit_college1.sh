#!/bin/bash

#SBATCH -J "LCR"
#SBATCH -p dlq
#SBATCH --exclude=compute[02-03]
#SBATCH -o /cm/home/cs14274101/archive/dj/log/lcr.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00

/cm/home/cs14274101/miniconda3/envs/py11/bin/python -u /cm/home/cs14274101/archive/dj/bca/train.py --cfg /cm/home/cs14274101/archive/dj/bca/cfgs/training/college1_LCR.toml
