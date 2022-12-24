#!/usr/bin/bash

#SBATCH -J yoloDNA
#SBATCH -D .
#SBATCH -o out_yoloDNA.txt
#SBATCH -e err_yoloDNA.txt
#SBATCH -n 60
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=coffea_genomes
#SBATCH --time=1-23:00:00
#SBATCH --mem=300G
####SBATCH --mail-type=ALL
####SBATCH --mail-user=estiven.valenciac@autonoma.edu.co

source ~/.bashrc
unset PYTHONPATH
conda activate YoloDNA

#INGRESE LOS INDICES DE LAS SECUENCIAS QUE DESEA EJECUTAR
for i in 188
do
    python3 pipelineDomain.py -t 0.5 -x $i
    rm -f *.fasta
done
