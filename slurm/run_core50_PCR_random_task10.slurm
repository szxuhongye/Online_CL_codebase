#!/bin/bash
#SBATCH -J PCR_core50_batch10_epoch50_random
#SBATCH -p tier3
#SBATCH --time=120:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=hx5239@rit.edu
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
cd /home/hx5239/PCR
nvidia-smi
python general_main.py --num_runs 10 --data core50 --cl_type nc --agent PCR  --retrieve random --update random --mem_size 10000 --epoch 50 --batch 10 --eps_mem_batch 10 --exp PCR_core50_epoch50_random_task10 --num_tasks 10