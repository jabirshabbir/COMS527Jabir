#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0-1:00:00  # max job runtime
#SBATCH --cpus-per-task=1  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=instruction  # partition(s)
#SBATCH --gres=gpu:2
#SBATCH --mem=16G  # max memory
#SBATCH -J "test527"  # job name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load ml-gpu/20230427 
# Modify the following path to your own work dir, which should be in /work/instruction/coms-527-f23/***
cd /work/instruction/coms-527-f23/jabir/DummyModel
ml-gpu python3 train.py
