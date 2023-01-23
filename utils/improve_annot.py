import pandas as pd
import numpy as np 
import sys
from Bio import SeqIO
from multiprocess import Process, Manager
import os 
import subprocess
from tqdm import tqdm
from Web_genome import download2

def renames_gff(df, original_domains):
    """
    Esta función convierte los nombres de los dominios del gff de anotación para que correspondan 
    con los nombres de los dominos del gff de predicción.

        df: Dataframe del gff.
        original_domains: Diccionario con los nomres de los dominios a convertir.

    """
    for i in original_domains.keys():
        df.replace(i,original_domains[i],inplace=True)

def find(name, path):
    """
    Esta función permite encontrar la ruta completa de los archivos gff de anotación.
        
        name: Nombre del genoma.
        path: Ruta padre donde se buscará el gff.
    """
    for root, dirs, files in os.walk(path, topdown=True):
        if name in files:
            return os.path.join(root, name)

def process(genome, df, chr, idx):
    file = open(f"tmp{idx}/{chr}",'w')
    ltr = df.LTR_ID.loc[df.Chromosome==chr].unique()
    #print(ltr)

    start = []
    end = []
    for i in ltr:
        division = i.split("_")
        #print(division)
        start.append(int(division[-2]))
        end.append(int(division[-1]))
    start.sort()
    end.sort()
    file.write(f">{chr}\n"+str(genome[chr].seq[:start[0]-1]+"N"*(end[0]-start[0]+1)))
    for i in range(1,len(start)):
        file.write(str(genome[chr].seq[end[i-1]:start[i]-1]+"N"*(end[i]-start[i]+1)))
    file.write(str(genome[chr].seq[int(end[-1]):])+"\n")
    file.close()

def main(genome, path_save, file_csv, idx, pIden, evalue, sensitive, name):
    fasta = SeqIO.to_dict(SeqIO.parse(genome, format='fasta'))
    df_genome = pd.read_csv(file_csv, sep=';')

    #Se determina cual es el genoma que se desea estudiar a partir del indice en el csv
    specie = df_genome['Species'].loc[idx-1]
    query = str(specie.replace(' ','_')+'.txt')
    path_query = find(query, path_anotation) #Ruta del archivo de anotación
    df_groundTrue = pd.read_csv(path_query, sep='\t')
    new_anot = df_groundTrue.copy()
    df_groundTrue.columns = [i.replace(' ','') for i in list(df_groundTrue.columns)]
    #print(df_groundTrue.LTR_ID.loc[df_groundTrue.Chromosome=="Chr10"].unique())
    #print(df_groundTrue)
    #sys.exit(0)
    chrs = df_groundTrue.Chromosome.unique()
    os.mkdir(f"tmp{idx}")
    jobs = []    
    for chr in range(len(chrs)):
        print("Empezando la cpu ",chr+1)
        pr = Process(target=process, args=(fasta, df_groundTrue, chrs[chr], idx))
        jobs.append(pr)
        pr.start()

    for job in jobs:
        job.join()

    masked_fasta = open(path_save+"/masked_"+name,"w")
    for i in fasta.keys():
        if i in chrs:
            with open(f"tmp{idx}/{i}","r") as f:
                for line in f:
                    masked_fasta.write(line)
        else:
            masked_fasta.write(f">{fasta[i].id}\n{fasta[i].seq}\n")
    masked_fasta.close()

    subprocess.run(f"rm -rf tmp{idx}", shell=True)
    command = f"diamond blastx -p 42 -k 1 -F 15 --range-culling -v --id {pIden} --evalue {evalue}{sensitive}-q {path_save}/masked_{name} -o {path_save}/{name}.m6 -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp scovhsp -d /shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/blastx_REXDB_GYPSYDB/GYREX.dmnd"
    subprocess.run(command, shell=True, check=True)
    df = pd.read_csv(f"{path_save}/{name}.m6", sep="\t", names=["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend","sstart","send","evalue","bitscore","qcovhsp","scovhsp"])
    for i in range(df.shape[0]):
        species = new_anot.loc[0,"Species"]
        LTR_ID = "."
        Chromosome = df.loc[i,"qseqid"]
        start = df.loc[i,"qstart"]
        end = df.loc[i,"qend"]
        domain = df.loc[i,"sseqid"].split("_")[0]
        length = abs(int(end)-int(start))
        superfamily = "."
        try:
            lineages = df.loc[i,"sseqid"].split("+")[-1]
        except:
            lineages = "."
        divergence = "."
        row = {"Species":species, "LTR_ID":LTR_ID, "Chromosome":Chromosome, "Start":start, "End":end, "Domain":domain, "Length(bp)":length, "Superfamily":superfamily,"Lineages":lineages, "Divergence":divergence}
        row = {i:[row[i]] for i in row.keys()}
        row = pd.DataFrame(row)
        new_anot = new_anot.append(row, ignore_index = True)
    new_anot.to_csv(path_save+"/annotation_news/"+name.replace(".fasta",".txt"), sep="\t", index=False)
    print(f"Tamano original de {name} con indice {idx}:\t",df_groundTrue.shape[0])
    print(f"Tamano nuevo de {name} con indice {idx}: \t",new_anot.shape[0])
    print(f"La diferencia es de: \t ",df.shape[0])

if __name__ == '__main__':
    """
    path = "/mnt/c/Users/estiv/Documents/Joven/Genomes/YoloDNA/metrics/"
    genome = "/mnt/c/Users/estiv/Documents/Joven/Genomes/Prueba_FP_blast/R498_Chr.fasta"
    #genome = "geno"
    masked_genome = "/mnt/c/Users/estiv/Documents/Joven/Genomes/Prueba_FP_blast/masked_genome.fasta"
    file_csv = path+"genomes_links.csv"
    path_anotation = path+"dataset_intact_LTR-RT"
    """
    
    path = "/shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/YoloDNA/metrics"
    file_csv = path+'/genomes_links.csv'
    path_anotation = path+'/dataset_intact_LTR-RT'
    path_save = "/shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/detect_ltr"
    pIden = 20
    evalue = 0.001
    sensitive = " --ultra-sensitive "
    idxs = [255,248,149,123,194,162,258,24,28,250,254,41]#[188,97,253,11,106,140,99,100,161]
    #idxs = [188,97,253,11,106,140,99,100,161]
    #idxs = [255]
    for idx in idxs:
        name = download2(file_csv, 100, path_save, idx, path_anotation, samples=-1)
        main(path_save+"/"+name, path_save, file_csv, idx, pIden, evalue, sensitive, name)
        os.remove(path_save+"/"+name)