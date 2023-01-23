import pandas as pd
import numpy as np
import os
import sys
import subprocess
from Bio import SeqIO
from multiprocess import Process
import subprocess
from Web_genome import download2

def filter_blast_report(path):
    f1 = open(path, "r")
    f = f1.readlines()
    f2 = open(path+"_correct","w")
    f2.write(f[0])
    for i in f[1:]:
        fil = i.split("\t")
        fil = [j for j in fil if j]
        if fil[1]==".":
            print(fil)
            fil[1] = f"{fil[2]}_{int(float(fil[3]))}_{int(float(fil[4]))}"
            f2.write("\t".join(fil))
        else:
            f2.write(i)
    f1.close()
    f2.close()
    subprocess.run(f"mv {path}_correct {path}", shell=True, check=True)

def masked(genome, df, chr, name):
    file = open(f"tmp_{name}/{chr}",'w')
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

def process(min, max, genome, lib, df):
    library = open(lib, 'a')
    for i in range(min, max):
        #print("i",i)
        domain = df.loc[i,"Domain"]
        seq = str(genome[df.loc[i,"Chromosome"]].seq[int(df.loc[i,"Start"])-1:int(df.loc[i,"End"])])
        library.write(f">Domain_{i}#{domain}\n")
        library.write(f"{seq}\n")
    library.close()



def filter_repeatmasker_report(path):
    file = open(path,"r").readlines()
    with open(path+"_filtered","w") as f:
        for row in file[3:]:
            items = [j for j in row.split(" ") if not j and (j not in ["+","C","-"])]
            items = "\t".join(items)
            f.write(items)
    return path+"_filtered"

def main(path, cpus, name, idx):

    genome = path+"/masked_"+name
    fasta = SeqIO.to_dict(SeqIO.parse(genome, format='fasta'))
    path2 = path+"/annotation_news/"+name.replace(".fasta",".txt")
    filter_blast_report(path2)
    df_groundTrue = pd.read_csv(path+"/annotation_news/"+name.replace(".fasta",".txt"), sep='\t')
    df_groundTrue.columns = [i.replace(' ','') for i in list(df_groundTrue.columns)]
    new_anot = df_groundTrue.copy()

    chrs = df_groundTrue.Chromosome.unique()
    os.mkdir(f"tmp_{name}")
    jobs = []    
    for chr in range(len(chrs)):
        print("Empezando la cpu ",chr+1)
        pr = Process(target=masked, args=(fasta, df_groundTrue, chrs[chr], name))
        jobs.append(pr)
        pr.start()

    for job in jobs:
        job.join()

    masked_fasta = open(path+"/masked_maskedblast_"+name,"w")
    for i in fasta.keys():
        if i in chrs:
            with open(f"tmp_{name}/{i}","r") as f:
                for line in f:
                    masked_fasta.write(line)
        else:
            masked_fasta.write(f">{fasta[i].id}\n{fasta[i].seq}\n")
    masked_fasta.close()
    subprocess.run(f"rm -rf tmp_{name}", shell=True)

    fasta = SeqIO.to_dict(SeqIO.parse(path+"/"+name, format='fasta'))
    len_process = int(df_groundTrue.shape[0]/(cpus-1))
    jobs = []
    #print(len_process)
    for cpu, pos in enumerate(range(len_process,df_groundTrue.shape[0],len_process)):
        print("Empezando la cpu ",cpu+1)
        min, max = pos-len_process, pos
        #print("Min max",min,max)
        pr = Process(target=process, args=(min, max, fasta, path+"/lib_"+name, df_groundTrue))
        jobs.append(pr)
        pr.start()
    print("Empezando la cpu ",cpus)
    min, max = pos, df_groundTrue.shape[0]
    pr = Process(target=process, args=(min, max, fasta, path+"/lib_"+name, df_groundTrue))
    jobs.append(pr)
    pr.start()

    for job in jobs:
        job.join()

    command = f'RepeatMasker {path}/masked_maskedblast_{name} -qq -lib {path}/lib_{name} -gff -nolow -dir {path}/repeat_{name.replace(".fasta","")} -pa {cpus}'
    subprocess.run(command, shell=True, check=True)
    path_report = name.replace(".fasta","")
    path_report = filter_repeatmasker_report(path=f"{path}/repeat_{path_report}/masked_maskedblast_{name}")
    
    df_masked = pd.read_csv(path_report, sep="/", names=["score","div","del","ins","sequence","qbegin","qend","left","repeat","class/family","rbegin","rend","left","ID"])
    
    for i in range(df_masked.shape[0]):
        species = new_anot.loc[0,"Species"]
        LTR_ID = "."
        Chromosome = df_masked.loc[i,"sequence"]
        start = df_masked.loc[i,"qbegin"]
        end = df_masked.loc[i,"qend"]
        domain = df_masked.loc[i,"class/family"].split("#")[-1]
        length = abs(int(end)-int(start))
        superfamily = "."
        lineages = "."
        divergence = "."
        if length < 1000:
            continue
        row = {"Species":species, "LTR_ID":LTR_ID, "Chromosome":Chromosome, "Start":start, "End":end, "Domain":domain, "Length(bp)":length, "Superfamily":superfamily,"Lineages":lineages, "Divergence":divergence}
        row = {i:[row[i]] for i in row.keys()}
        row = pd.DataFrame(row)
        new_anot = new_anot.append(row, ignore_index = True)
    new_anot.to_csv(path+"/annotation_news/repeat_"+name.replace(".fasta",".txt"), sep="\t", index=False)
    print(f"Tamano original de {name} con indice {idx}:\t",df_groundTrue.shape[0])
    print(f"Tamano nuevo de {name} con indice {idx}: \t",new_anot.shape[0])
    print(f"La diferencia es de: \t ",df_masked.shape[0])



if __name__ == '__main__':
    """
    path = "/mnt/c/Users/estiv/Documents/Joven/Genomes/YoloDNA/metrics/"
    genome = "/mnt/c/Users/estiv/Documents/Joven/Genomes/Prueba_FP_blast/R498_Chr.fasta"
    #genome = "geno"
    lib = "/mnt/c/Users/estiv/Documents/Joven/Genomes/Prueba_FP_blast/lib.fasta"
    file_csv = path+"genomes_links.csv"
    path_anotation = path+"dataset_intact_LTR-RT"
    idx = 188
    cpus = 12
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
    idxs = [188]
    cpus = 42
    #annots = os.listdir(path_save+"/annotation_news")
    for idx in idxs:
        #name = download2(file_csv, 500, path_save, idx, path_anotation, samples=-1)
        name="Oryza_sativa_ssp._Indica.fasta"
        main(path_save, cpus, name, idx)
