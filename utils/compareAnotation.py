from optparse import OptionParser
import numpy as np
import pandas as pd
import os
import sys
from Bio import SeqIO
from utils.extra_functions import clustering
from utils.metrics_after_LTR_harvest import metrii

def find(name, path):
    """
    Esta función permite encontrar la ruta completa de los archivos gff de anotación.
        
        name: Nombre del genoma.
        path: Ruta padre donde se buscará el gff.
    """
    for root, dirs, files in os.walk(path, topdown=True):
        if name in files:
            return os.path.join(root, name)

def extract_ids(groundT, pred):
    """
    Esta función extrae los ids de las secuencias de los gff de 
    anotación y de predicción. Y devuelve los ids que son unicos a ambos gff, 
    y los que son unicos a cada gff.

        ground: Dataframe que contiene el gff de anotación.
        pred: Dataframe que contiene el gff de predicción.
    """
    ids_pred = pred.index.unique().tolist()
    ids_ground = groundT.index.unique().tolist()
    ids_union = ids_pred and ids_ground
    ids_pred_unique = list(set(ids_pred) - set(ids_union))
    ids_ground_unique = list(set(ids_ground) - set(ids_union))
    return ids_pred_unique, ids_ground_unique, ids_union

def renames_gff(df, original_domains):
    """
    Esta función convierte los nombres de los dominios del gff de anotación para que correspondan 
    con los nombres de los dominos del gff de predicción.

        df: Dataframe del gff.
        original_domains: Diccionario con los nomres de los dominios a convertir.

    """
    for i in original_domains.keys():
        df.replace(i,original_domains[i],inplace=True)

def metrics(df_pred, file, path_query):
    """
    Esta función calcula las métricas de precisión, recall, f1 y exactitud para los archivos 
    gff de anotación y predicción.
    
        threshold: Valor umbral de precensia con el que se determina el NMS.
        df_groundT: Dataframe del gff de anotación.
        df_pred: Dataframe del gff de predicción.
        union_ids: Array con los ids de las secuencias presentes en ambos archivos.
        pred_ids: Array con los ids de las secuencias presentes solo en el archivo de predicción. 
        ground_ids: Array con los ids de las secuencias presentes solo en el archivo de anotación.
        longNormal: Diccionario con las longitudes de los dominios.
        classes: Diccionario con las classes de los dominios.
        dicc_size: Diccionario con las longitudes de los dominos. 
    """
    parser = list(SeqIO.parse(file, 'fasta'))
    if ' ' in parser[0].id:
      genoma = [(x.id.split(' ')[0], x.seq) for x in parser]
    else:
      genoma = [(x.id, x.seq) for x in parser]
    genoma = {x:y for (x,y) in genoma}
    new_file = file+'potential_TE'
    resultado = open(new_file,'w')
    ext = 8000
    dist_max = 4000
    dom_min = 3
    for id in df_pred.index.unique().tolist():
        Y_pred = df_pred.loc[id].copy()
        yy_start = np.array(Y_pred["Start"])
        yy_end = yy_start + np.array(Y_pred["Length"])
        if yy_start.size ==1:
          yy_start = np.array([yy_start])
          yy_end = np.array([yy_end])
        indices2 = [(yy_start[i],yy_end[i]) for i in range(len(yy_start))]
        doms = clustering(A=indices2, dist_max=dist_max, dom_min=dom_min)
        for tupla in doms:
          if tupla[0]-ext<0:
            inicio = 0
          else:
            inicio = tupla[0]-ext
          if tupla[1]+ext>=len(genoma[id]):
            fin = len(genoma[id])
          else:
            fin = tupla[1]+ext
          resultado.write(f">{id}-{inicio}-{fin}\n")
          resultado.write(f"{genoma[id][inicio:fin]}\n")
    return metrii(file, new_file, genoma, path_query)

def analysis(file_csv, path_anotation, idx, path_pred, name_file, threshold, inpactorTest,file_fasta):
    original_domains = {'RH':'RNASEH','aRH':'RNASEH','RH/aRH':'RNASEH','PROT':'AP','intact_5ltr':'LTR','intact_3ltr':'LTR'}
    #Se lee el archivo .tab generado por la red
    df_pred = pd.read_csv(path_pred, sep='\t').replace({' ':''}, regex=True)
    df_pred.columns = [i.replace(' ','').replace('|','') for i in list(df_pred.columns)]
    df_pred.sort_values(['id','Start'] , inplace = True)
    df_pred.set_index('id', drop=True, inplace=True)
    df_pred = df_pred.loc[
        (df_pred['Class']=='LTR')|
        (df_pred['Class']=='GAG')|
        (df_pred['Class']=='RT')|
        (df_pred['Class']=='RNASEH')|
        (df_pred['Class']=='INT')|
        (df_pred['Class']=='AP')
    ]
    df_pred[['ProbabilityPresence','ProbabilityClass']] = df_pred[['ProbabilityPresence','ProbabilityClass']].astype(np.float32)
    #Se abre un archivo donde se guardarán las métricas
    file = open(name_file, 'w')
    path_query = None
    if idx != None:
        #Se lee el archivo csv con los genomas
        df_genome = pd.read_csv(file_csv, sep=';')

        #Se determina cual es el genoma que se desea estudiar a partir del indice en el csv
        specie = df_genome['Species'].loc[idx-1]
        query = str(specie.replace(' ','_')+'.txt')
        path_query = find(query, path_anotation) #Ruta del archivo de anotación

        if path_query is not None:
            #Se lee el archivo de anotación de la especie
            df_groundTrue = pd.read_csv(path_query, sep='\t')
            df_groundTrue.columns = [i.replace(' ','') for i in list(df_groundTrue.columns)]
            df_groundTrue.set_index('Chromosome', drop=True, inplace=True)
            renames_gff(df_groundTrue, original_domains)
            df_groundTrue = df_groundTrue.loc[
                (df_groundTrue['Domain']=='LTR')|
                (df_groundTrue['Domain']=='GAG')|
                (df_groundTrue['Domain']=='RT')|
                (df_groundTrue['Domain']=='RNASEH')|
                (df_groundTrue['Domain']=='INT')|
                (df_groundTrue['Domain']=='AP')
            ]
            pred_ids,ground_ids,union_ids = extract_ids(groundT=df_groundTrue, pred=df_pred)  
            file.write(f'Analysis for the specie of {specie}\n\n')
            file.write(f'Numero de ids unicos para el archivo de prediccion: {len(pred_ids)}\n')
            file.write(f'Numero de ids unicos para el archivo de ground True: {len(ground_ids)}\n')
            file.write(f'Numero de ids que se comparten en los dos archivos: {len(union_ids)}\n\n')
        else:
            print('ERROR: no se encontro la ruta del archivo de anotacion')
            sys.exit(1)
    Precision, Recall, Accuracy, F1 = metrics(df_pred,file_fasta,path_query)
    file.write(f"Precision: {Precision} \nRecall: {Recall}\nAccuracy: {Accuracy}\nF1: {F1}\n")
