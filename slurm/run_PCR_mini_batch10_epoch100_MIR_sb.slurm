#!/bin/bash
#SBATCH -J PCR_mini_imagenet_batch10_epoch100_MIR_non_sb_mem20000
#SBATCH -p tier3
#SBATCH --time=96:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=hx5239@rit.edu
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
cd /home/hx5239/PCR
nvidia-smi
python general_main.py --num_runs 10 --data mini_imagenet --cl_type nc --agent PCR  --retrieve MIR --update random --mem_size 10000 --epoch 100 --batch 10 \
--eps_mem_batch 8 --sub_eps_mem_batch 2 --exp PCR_mini_imagenet_batch10_epoch100_MIR_non_sb_mem20000 --second_buffer True --retrieve2 IMIR --update2 random