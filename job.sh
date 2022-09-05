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
for i in 85 187 189
do
    conda deactivate
    conda activate YoloDNA

    python3 Web_genome.py -c "metrics/genomes_links.csv" -T 500 -i $i
    filename=`ls *.fasta`
    flag=,`cut -f1 ERROR.txt`
    if [ "$flag" == "TRUE" ]
    then
        rm -f ERROR.txt
        echo "La secuencia $filename fallo"
        rm *.fasta
        rm *.zip
    else
        chmod +777 *.fasta
        rm -f ERROR.txt
        size=`du -sh "$filename" | cut -f1`
        echo "El tamano del genoma $filename es: $size"
        conda deactivate
        conda activate YoloDNA2

        echo "Ejecutando el genoma ${filename} secuencia $i"
        python3 pipelineDomain.py -f $filename -o ${filename/fasta/tab} -t 0.5 -x $i
        rm -f $filename
    fi
done
