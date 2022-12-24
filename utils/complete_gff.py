from optparse import OptionParser
import random 
import numpy as np 
import pandas as pd 
random.seed(8)
import os
import sys
from Bio import SeqIO
from tensorflow.keras import backend as K 
import time
from tqdm import tqdm
import subprocess

def IOU(box1,box2,size1,size2):
    pi1,len1,n1 = box1
    pi2,len2,n2 = box2

    pi1=(pi1+n1)*100
    pf1=pi1+len1*size1
    pi2=(pi2+n2)*100
    pf2=pi2+len2*size2
    xi1 = max([pi1,pi2])
    xi2 = min([pf1,pf2])
    inter_width = xi2-xi1
    inter_area = max([inter_width,0])
    box1_area = len1*size1
    box2_area = len2*size2
    union_area = box1_area+box2_area-inter_area
    iou = inter_area/(union_area + K.epsilon())
    return iou

def NMS(Yhat, threshold_presence, threshold_NMS, dicc_size):
    mascara = (Yhat[:,0:1]>=threshold_presence)*1
    data_pred = mascara*Yhat
    data_mod = np.copy(data_pred[:,0])
    cont=1
    while cont>0:
        try:
            ind_first = np.nonzero(data_mod)[0][0]
        except:
            break
        ind_nonzero = np.nonzero(data_mod)[0][1:]
        for i in ind_nonzero:
            box1=[data_pred[ind_first,1],data_pred[ind_first,2],ind_first]
            box2=[data_pred[i,1],data_pred[i,2],i]
            size1=dicc_size[np.argmax(data_pred[ind_first,3:])]
            size2=dicc_size[np.argmax(data_pred[i,3:])]
            iou = IOU(box1,box2,size1,size2)
            if iou>=threshold_NMS:
                if data_mod[i]>data_mod[ind_first]:
                    data_pred[ind_first,:]=0
                    data_mod[ind_first]=0
                    break
                else:
                    data_pred[i,:]=0
                    data_mod[i]=0
            else:
                data_mod[ind_first]=0
                break
        cont=np.sum(ind_nonzero)
    return data_pred

def find(name, path):
    """
    Esta función permite encontrar la ruta completa de los archivos gff de anotación.
        
        name: Nombre del genoma.
        path: Ruta padre donde se buscará el gff.
    """
    for root, dirs, files in os.walk(path, topdown=True):
        #print(files)
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

def positions(df_init,df_end, id):
    """
    Esta función retorna los arrays de las posiciones de inicio y fin 
    de los dominios en una secuencia en especifico.

        df_init: Dataframe que contiene las posiciones de inicio de la secuencia
        df_end: Dataframe que contiene las posiciones de fin de la secuencia.
    
    """
    try:
        if isinstance(df_init.loc[id].tolist(), list): 
            start = np.array([int(i) for i in df_init.loc[id].tolist()])
            end = np.array([int(i)+j for i,j in zip(list(df_end.loc[id].tolist()),start)])
            #print(f'id {id} ---------- {df_init.loc[id].tolist()}')
        else:
            start = np.array([int(df_init.loc[id])])
            end = np.array([int(df_end.loc[id])+start[0]])
        return start, end
    except:
        return None, None

def nucleotide(start, end, array_start=[], array_end=[]):
    """
    Esta función toma las posiciones de principio y fin del dominio y construye la longitud del
    dominio correspondiente. 

        start: Posición de inicio del dominio más próximo en la secuencia.
        end: Posición de fin del dominio más lejano en la secuencia.
        array_start: Array que contiene las posiciones de inicio de todos los 
        dominios en la secuencia.
        array_end: Array que contiene las posiciones de fin de todos los dominios en la
        secuencia. 

    """
    vect = np.zeros((end-start), dtype=bool)
    for dom_init, dom_end in zip(array_start, array_end):
        vect[dom_init-start:dom_end-start+1] = 1
    return vect

def nt_TE_LTR(y,ventana,threshold_presence, dicc_size):
    """
    Esta función contruye las logitudes de los dominios LTR en la secuencia.

        Y: Array con las predicción de los dominios en la secuencia.
        ventana: Longitud de la secuencia.
        threshold_presence: Umbral de presencia.
        dicc_size: Diccionario con los valores de lontigud de los dominios.
    """
    nucleotidos = np.zeros((ventana))
    #mask = (y[:,0:1]>=threshold_presence)*1
    #valores =[]
    indices = np.nonzero(y[:,9])[0]
    if len(indices)%2==0:
        for h in range(0,len(indices),2):
            size = dicc_size[6]
            j1=indices[h]
            j2=indices[h+1]
            inicio = int(j1*100+y[j1,1]*100)
            inicio2=int(j2*100+y[j2,1]*100)
            fin = int(inicio2+y[j2,2]*size)
            nucleotidos[inicio:fin]=1
    return nucleotidos

def revert_gff(start, length, domains, max_size, longNormal, classes):
    """
    Esta función revierte la anotación de los gff en arrays similares a la salida de la red.

        start: Dataframe con las posiciones de inicio de los dominios de la secuencia.
        length: Dataframe con las longitudes de los dominios de la secuencia.
        domains: Clasificación de los dominios presentes en la secuencia.
        max_size: Posición final del dominio más alejado en la secunecia.
        longNormal: Diccionario con las longitudes de los dominios.
        classes: Diccionario con las clases de los dominios.
    
    """
    #start = start-min_size
    array = np.zeros((int(max_size/100),10))
    if start.size==1:
        #print(f'l {length} dom {str(domains)}')
        start_value = int(start/100)
        array[start_value,0] = 1
        array[start_value,1] = (start-int(start/100)*100)/100
        array[start_value,2] = length/longNormal[classes[str(domains)]]
        array[start_value,classes[str(domains)]] = 1
        return array
    else:
        for i in range(start.shape[0]):
            start_value = int(start[i]/100)
            array[start_value,0] = 1
            array[start_value,1] = (start[i]-int(start[i]/100)*100)/100
            array[start_value,2] = length[i]/longNormal[classes[domains[i]]]
            array[start_value,classes[domains[i]]] = 1
        return array
    
def get_index(Y_true, Y_pred, threshold, dicc_size, file_index, id):
    Y1_index = list(np.where(Y_true[:,0]==1)[0])
    Y2_index = list(np.where(Y_pred[:,0]==1)[0])
    dom_union1 = []
    dom_union2 = []
    for i in Y1_index:
        flag=False
        for k,j in enumerate(Y2_index):
            box1=[Y_true[i,1], Y_true[i,2],i]
            box2=[Y_pred[j,1],Y_pred[j,2],j]
            dom1=np.argmax(Y_true[i,3:])
            dom2=np.argmax(Y_pred[j,3:])
            size1=dicc_size[dom1]
            size2=dicc_size[dom2]
            iou = IOU(box1,box2,size1,size2)
            if iou>=threshold:
                dom_union1.append(i)
                dom_union2.append(j)
                pos = k
                flag = True
                break
        if flag:
            del Y2_index[pos]
        if not Y2_index:
            break
    dom_unique = np.array(Y2_index)
    file_index.write(f"index para id: {id}\n {str(dom_unique)}\n")
    print("Tamaño de unicos",dom_unique)
    return dom_unique, dom_union1, dom_union2

def get_fasta(Y, ids, seq, file, id, classes, k, dicc_size):
    for i in ids:
        start = int((i+1+Y[i,1])*100)
        size= dicc_size[np.argmax(Y[i,3:])]
        dom = classes[np.argmax(Y[i,3:])+3]
        end = int(start+Y[i,2]*size)
        #print(seq_final)
        #print(seq[start:end])
        k += 1
        file.write(f">domain{k}={dom}_seqid_{id}_start_{start}_end_{end}\n")
        file.write(str(seq[start:end])+"\n")
    return k

def renames_gff(df, original_domains):
    """
    Esta función convierte los nombres de los dominios del gff de anotación para que correspondan 
    con los nombres de los dominos del gff de predicción.

        df: Dataframe del gff.
        original_domains: Diccionario con los nomres de los dominios a convertir.

    """
    for i in original_domains.keys():
        df.replace(i,original_domains[i],inplace=True)

def metrics(df_groundT, df_pred, union_ids, pred_ids, ground_ids, longNormal, classes, dicc_size, dict_seq, outPath):
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
    file_unique = open(outPath+'/FP.fasta','w')
    file_union1 = open(outPath+'/TP.fasta','w')
    file_union2 = open(outPath+'/TP.fasta','w')
    file_index = open(outPath+'index_FP.txt','w')
    k1 = 0; k2 = 0; k3 = 0 
    for id in tqdm(union_ids):
        Y_ground = df_groundT.loc[id].copy()
        Y_pred = df_pred.loc[id].copy()
        max_true, max_pred = Y_ground['Length(bp)'].max()+Y_ground['Start'].max(), Y_pred['Length'].max()+Y_pred['Start'].max()
        max_size = int(max(max_true, max_pred))
        Y_true = revert_gff(np.array(Y_ground["Start"]), np.array(Y_ground["Length(bp)"]), np.array(Y_ground["Domain"]), max_size, longNormal, classes)
        
        Y_pred = revert_gff(np.array(Y_pred["Start"]), np.array(Y_pred["Length"]), np.array(Y_pred["Class"]), max_size, longNormal, classes)
        dom_unique, dom_union1, dom_union2 = get_index(Y_true, Y_pred, 0.6, dicc_size, file_index, id)
        classes2 = {3:"RT",4:"GAG",5:"ENV",6:"INT",7:"AP",8:"RNASEH",9:"LTR"}
        k1 = get_fasta(Y_pred, dom_unique, dict_seq[id].seq, file_unique,id,classes2, k1, dicc_size)
        k2 = get_fasta(Y_pred, dom_union2, dict_seq[id].seq, file_union1,id,classes2, k2, dicc_size)
        k3 = get_fasta(Y_true, dom_union1, dict_seq[id].seq, file_union2,id,classes2, k3, dicc_size)   
    file_unique.close()
    file_union1.close()
    file_union2.close()

    command = f"""diamond blastx -k 1 -v --ultra-sensitive --id 80 --query-cover 80 -q {outPath}/FP.fasta 
                -d /shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/blastx_REXDB_GYPSYDB/GYREX.dmnd -p 48 -o {outPath}/Blast_FP.m6 
                -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp scovhsp"""
    subprocess.run(command, shell=True, check=True)
    df = pd.read_csv(outPath+"/Blast_FP.m6", sep="\t", names=["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend","sstart","send","evalue","bitscore","qcovhsp","scovhsp"])
    
    file = pd.read_csv




def main(file_csv, path_anotation, idx, path_pred, genome, path_complete_gff):
    dicc_size={0:1000,1:3000,2:2000,3:2000,4:1000,5:1000,6:10000}
    classes = {"RT":3,"GAG":4,"ENV":5,"INT":6,"AP":7,"RNASEH":8,"LTR":9}
    longNormal = {3:1000,4:3000,5:2000,6:2000,7:1000,8:1000,9:10000}
    original_domains = {'RH':'RNASEH','aRH':'RNASEH','RH/aRH':'RNASEH','PROT':'AP','intact_5ltr':'LTR','intact_3ltr':'LTR'}

    #Se lee el archivo csv con los genomas
    df_genome = pd.read_csv(file_csv, sep=';')

    #Se determina cual es el genoma que se desea estudiar a partir del indice en el csv
    specie = df_genome['Species'].loc[idx-1]
    query = str(specie.replace(' ','_')+'.txt')
    path_query = find(query, path_anotation) #Ruta del archivo de anotación

    #Se crea carpeta para el gff que se va a modificar
    outPath = path_complete_gff+"/"+specie.replace(' ','_') 
    if not os.path.exists(outPath):
        os.mkdir(outPath)

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
    else:
        print('ERROR: no se encontro la ruta del archivo de anotacion')
        sys.exit(1)

    #Se lee el archivo .tab generado por la red
    df_pred = pd.read_csv(path_pred, sep='\t').replace({' ':''}, regex=True)
    df_pred.columns = [i.replace(' ','').replace('|','') for i in list(df_pred.columns)]
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
    
    pred_ids, ground_ids, union_ids = extract_ids(groundT=df_groundTrue, pred=df_pred)  
    #print('Conteo IDS predichos unicos: ',len(pred_ids))
    #print('Conteo de IDS unicos: ',len(union_ids))
    #print('Conteo de IDS verdaderos unicos: ',len(ground_ids))
    
    dict_seq = SeqIO.to_dict(SeqIO.parse(genome, 'fasta'))
    metrics(df_groundTrue, df_pred, union_ids, pred_ids, ground_ids, longNormal, classes, dicc_size, dict_seq, outPath)

if __name__ == '__main__':
    path_anotation = '../../YoloDNA/metrics/dataset_intact_LTR-RT'
    pred_gff = "Oryza_sativa_ssp._Indica.tab"
    genome = "Oryza_sativa_ssp._Indica.fasta"
    main(file_csv, path_anotation, idx, path_pred, genome)