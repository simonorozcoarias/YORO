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

for i in {1..1}
do
    echo "Descargando secuencia $i"
    
    conda deactivate
    conda activate YoloDNA
    
    python3 Web_genome.py -c "metrics/genomes_links.csv" -T 100 -i $i
    filename=`ls *.fasta`
    flag=`cut -f1 ERROR.txt`
        if [ "$flag" == "TRUE" ] 
                then
                rm -f ERROR.txt
                break
                fi
    rm -f ERROR.txt

    conda deactivate
    conda activate YoloDNA2

    echo "Ejecutando el genome ${filename} secuencia $i"
    python3 pipelineDomain.py -f $filename -o ${filename/fasta/tab} -t 0.5 -x $i
    rm -f $filename
done