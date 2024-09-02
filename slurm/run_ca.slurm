#!/bin/bash
#SBATCH -J ca_infoNCE_epoch50_ASER
#SBATCH -p tier3
#SBATCH --time=48:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=hx5239@rit.edu
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
cd /home/dir
nvidia-smi
python general_main.py --num_runs 10 --data cifar100 --cl_type nc --agent PCR_ca  --retrieve ASER --update ASER --mem_size 1000 --epoch 50 --batch 10 --eps_mem_batch 10 --exp ca_infoNCE_epoch50_ASER