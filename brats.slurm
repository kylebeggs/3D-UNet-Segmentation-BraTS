#!/bin/bash

#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-gpu=10G
#SBATCH --time=100:00:00
#SBATCH --job-name=BRATS

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user kbeggs07@knights.ucf.edu

# Load modules
module load anaconda/anaconda3
module list
source activate torch-medical-seg

# check gpu allocation
nvidia-smi
nvidia-smi topo -m

# run model
python train.py --name brats --epochs 100 --lr 0.001
