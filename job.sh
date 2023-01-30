#!/usr/bin/bash

#SBATCH -J yolotests
#SBATCH -D .
#SBATCH -o out_yolotest.txt
#SBATCH -e err_yolotest.txt
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=coffea_genomes
#SBATCH --mem=300G

source ~/.bashrc
unset PYTHONPATH
conda activate YoloDNA

python TestExecution.py 
