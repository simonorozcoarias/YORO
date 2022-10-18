#!/usr/bin/python3
from optparse import OptionParser 
import random 
import numpy as np 
import pandas as pd 
import requests
random.seed(8)
from zipfile import ZipFile
import os
import gzip
import shutil
import gdown
from Bio import SeqIO
from tqdm import tqdm
import sys
import subprocess

def find(name, path):
    for root, dirs, files in os.walk(path, topdown=True):
        if name in files:
            return os.path.join(root, name)

#Las paginas de los genomas son muy diferentes teniendo diferentes formatos y donde en la mayoria 
#los links no dirigen al genoma sino a los bioprojects
def unique_webpage(df_genome, count=False):
    """
    Retorna una lista con los enlaces unicos que existen en la base de datos
    df_genome: dataFrame con la base de datos

    count: bandera que permite imprimir la cantidad de links para cada una de las bases
    """
    links = df_genome['Data sources'].tolist()
    pages = []
    for i,j in enumerate(links):
        pages.append(j.split('/')[2])
    pages = list(set(pages))
    if count:
        print("Unique pages: ")
        for j in pages:
            print(j,': ',df_genome["Data sources"].loc[df_genome["Data sources"].str.contains(j, case=False)].count())
    return pages

def download(df_genome,path_save,sample,timeout,index):
    if sample != -1:
        sample = random.sample(range(301),sample)
        links = df_genome['Filtered data'].loc[sample].tolist()
        [names.replace(' ', '_')+'.fasta' for names in df_genome['Species'].loc[sample].tolist()]
    else: 
        links = [df_genome['Filtered data'].loc[index-1]]
        names = [df_genome['Species'].loc[index-1].replace(' ','_')+'.fasta']
    for i,j in tqdm(enumerate(links)):
        print('descargando ',names[i])
        if j.split(' ')[0] == 'curl':
            name_file = names[i].split('.')[0]
            curl_link = j.split(' ')[10]
            cook_part = j.split(' ')[2].split('=')
            cookies = {cook_part[0]:cook_part[1]}
            data_part = j.split(' ')[6].replace('\\"','').replace('{','').replace('[','').replace(']}}','').replace('"','').split(':')
            data={str(data_part[0]):{str(data_part[1]):[str(data_part[2])]}}
            try:
                response = requests.post(curl_link, cookies=cookies, json=data, verify=False, timeout=timeout)
                with open(path_save+'/'+name_file+'.zip', 'wb') as f:
                    f.write(response.content)
                with ZipFile(path_save+'/'+name_file+'.zip','r') as zip:
                    zip.extractall(path_save)
                try:
                    os.remove(path_save+'/'+name_file+'.zip')
                except:
                    pass
                genome = find('.gz',path_save)
                if genome is not None:
                    with gzip.open(genome,'rb') as f_1:
                        name_2 = names[i].replace('\n','')
                        with open(path_save+'/'+name_2,'wb') as f_2:
                            shutil.copyfileobj(f_1,f_2)
                    shutil.rmtree(path_save+'/'+genome.split('\\')[1])
                    if os.path.isfile(path_save+'/'+name_file+'.zip'):
                        os.remove(path_save+'/'+name_file+'.zip')
                    if os.path.isfile(path_save+'/'+names[i]+'.gz'):
                        os.remove(path_save+'/'+names[i]+'.gz')
                        return names
                    else:
                        print('El archivo .gz no se elimino')
                else: 
                    print('No se pudo descomprimir el archivo: ',name_file)
            except:
                print('Fallo la descarga de: ',name_file)
        elif j.split('/')[2] in 'docs.google.com':
            print(j)
            try:
                gdown.download(j,path_save+'/'+names[i]+'.gz', quiet=False)
                with gzip.open(path_save+'/'+names[i]+'.gz','rb') as f_1:
                    with open(path_save+'/'+names[i],'wb') as f_2:
                        shutil.copyfileobj(f_1,f_2)
                if os.path.isfile(path_save+'/'+names[i]+'.gz'):
                    os.remove(path_save+'/'+names[i]+'.gz')
                    return names
                else:
                    print('El archivo .gz no se elimino')
            except: 
                print('Fallo la descarga de ',names[i])
        else:
            if '.gz' in j.split('/')[-1]:  
                try:
                    respose = requests.get(j, timeout=timeout)
                    open(path_save+'/'+names[i]+'.gz','wb').write(respose.content)
                    with gzip.open(path_save+'/'+names[i]+'.gz','rb') as f_1:
                        with open(path_save+'/'+names[i],'wb') as f_2:
                            shutil.copyfileobj(f_1,f_2)
                    if os.path.isfile(path_save+'/'+names[i]+'.gz'):
                        os.remove(path_save+'/'+names[i]+'.gz')
                        return names
                    else:
                        print('El archivo .gz no se elimino')
                except:
                    print('Fallo la descarga de ',names[i])
            elif '.fa' or '.fasta' in j.split('/')[-1]:
                try:
                    respose = requests.get(j, timeout=timeout)
                    open(path_save+'/'+names[i],'wb').write(respose.content)
                    return names
                except:
                    print('Fallo la descarga de ',names[i])
    return None

def download2(file_csv, timeout, path_save, idx, path_anotation, samples=-1):
    df_genome = pd.read_csv(file_csv, sep=';')
    links_unique = unique_webpage(df_genome, count=False)
    names = download(df_genome,path_save,sample=samples,timeout=timeout,index=idx)
    path_query = find(names[0].replace('fasta','txt'), path_anotation)
    
    #Validación de si el archivo descargado coincide con el de anotación
    if names is not None:
        if path_query is not None:
            df_anot = pd.read_csv(path_query, sep='\t')
            df_anot.columns = [i.replace(' ','') for i in list(df_anot.columns)]
            id_anot = df_anot['Chromosome'].unique().tolist()
            seq = list(SeqIO.parse(path_save+'/'+names[0], 'fasta'))
            id_fasta = [i.id for i in seq]
            #Se comprueba que al menos la ventiaba parte de los indices del archivo de anotación correspondan al archio fasta
            check = random.sample(id_anot, int(len(id_anot)/5))
            for j in check:
                if j not in id_fasta:
                    print('La secuencia ',names[0],' con indice',idx,' no coincide con el archivo de anotación')
                    sys.exit("ERROR: los id entre el archivo de anotación y el genoma no son iguales")
            print('La secuencia ',names[0],' con indice',idx,' coincide con el archivo de anotación')
        else:
            print('La secuencia ',names[0],' no con indice',idx,' coincide con el archivo de anotación')
            sys.exit(1)
    return names[0]

