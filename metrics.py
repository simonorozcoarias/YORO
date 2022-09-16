#!/usr/bin/python3
from multiprocessing import parent_process
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
from Bio import SeqIO
#import gdown
#from tqdm import tqdm
import sys
import time
from tensorflow.keras import backend as K 
from tqdm import tqdm

#NO PONER CUIDADO
file_csv = 'genomes_links.csv'
idx = 1
name_file = 'analysis_Inpatorv2.out'
folder = '_'
domain = 'all'
threshold = [0.74]
#th_class = 0.95

#Solo este archivo
path_pred = '~/Documents/GitHub/output.tab'

dicc_size={0:1000,1:3000,2:2000,3:2000,4:1000,5:1000,6:10000}

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

def NMS(Yhat, threshold_presence, threshold_NMS):
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
    for root, dirs, files in os.walk(path, topdown=True):
        if name in files:
            return os.path.join(root, name)

def extract_ids(groundT, pred):
    ids_pred = pred.index.unique().tolist()
    ids_ground = groundT.index.unique().tolist()
    ids_union = ids_pred and ids_ground
    ids_pred_unique = list(set(ids_pred) - set(ids_union))
    ids_ground_unique = list(set(ids_ground) - set(ids_union))
    return ids_pred_unique, ids_ground_unique, ids_union
"""
def positions(df_init,df_end, id):
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
    vect = np.zeros((end-start), dtype=bool)
    for dom_init, dom_end in zip(array_start, array_end):
        vect[dom_init-start:dom_end-start+1] = 1
    return vect
"""
def nt_TE(y,ventana,threshold_presence,distancia=30):
    nucleotidos = np.zeros((ventana))
    mask = (y[:,0:1]>=threshold_presence)*1
    valores =[]
    indices = np.nonzero(mask[:,0])[0]
    for h in range(len(indices)):
        size = dicc_size[np.nonzero(y[indices[h],3:]==np.amax(y[indices[h],3:]))[0][0]]
        if h!=0 and (indices[h]-indices[h-1])<distancia:
            j1=indices[h-1]
            j2=indices[h]
            inicio = int(j1*100+y[j1,1]*100)
            inicio2=int(j2*100+y[j2,1]*100)
            fin = int(inicio2+y[j2,2]*size)
            nucleotidos[inicio:fin]=1
    return nucleotidos

def nt_TE_LTR(y,ventana,threshold_presence):
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


classes = {"RT":3,"GAG":4,"ENV":5,"INT":6,"AP":7,"RNASEH":8,"LTR":9}
longNormal = {3:1000,4:3000,5:2000,6:2000,7:1000,8:1000,9:10000}

def revert_gff(start, length, domains, min_size, max_size):
    #start = start-min_size
    array = np.zeros((500,10))
    for i in range(len(start)):
        start_value = int(start[i]/100)
        array[start_value,0] = 1
        array[start_value,1] = (start[i]-start_value*100)/100
        array[start_value,2] = length[i]/longNormal[classes[domains[i]]]
        array[start_value,classes[domains[i]]] = 1
    return array

def metrics(threshold, df_groundT, df_pred, ids):
    TP = 0; FP = 0; FN = 0; TN=0
    for id in ids:
        Y_ground = df_groundT.loc[id].copy()
        Y_pred = df_pred.loc[id].copy()
        min_true, min_pred = Y_ground['Start'].min(), Y_pred['Start'].min()
        max_true, max_pred = Y_ground['Length'].max()+Y_ground['Start'].max(), Y_pred['Length'].max()+Y_pred['Start'].max()

        min_size = min(min_true, min_pred)
        max_size = max(max_true, max_pred)

        Y_true = revert_gff(np.array(Y_ground["Start"]), np.array(Y_ground["Length"]), np.array(Y_ground["Domain"]), min_size, max_size)
        Y_pred = revert_gff(np.array(Y_pred["Start"]), np.array(Y_pred["Length"]), np.array(Y_pred["Class"]), min_size, max_size)
        Y_pred = NMS(Y_pred, threshold, 0.1)

        Y_true_nt = nt_TE_LTR(Y_true,50000,threshold)
        Y_pred_nt = nt_TE_LTR(Y_pred,50000,threshold)
        TP_window = np.sum(Y_true_nt*Y_pred_nt)
        TP += TP_window
        FP += np.sum(Y_pred_nt)-TP_window
        FN += np.sum(Y_true_nt)-TP_window
        TN += Y_true_nt.shape[0]-TP_window-(np.sum(Y_pred_nt)-TP_window)-(np.sum(Y_true_nt)-TP_window)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Accuracy = (TP + TN)/ (TP + FN + TN + FP)
    F1 = 2* Precision*Recall/(Precision + Recall)
    print(f"Precision: {Precision} \nRecall: {Recall}\nAccuracy: {Accuracy}\nF1: {F1}")
        #Y_true = revert_gff
        #Y_pred = 

def main():
    begin = time.time()
    path_query = 'archivo_fastTEST_con_LTR.gff'
    df_groundTrue = pd.read_csv(path_query, sep='\t').replace({' ':''}, regex=True)
    df_groundTrue.columns = [i.replace(' ','').replace('|','') for i in list(df_groundTrue.columns)]
    df_groundTrue.set_index('id_secuence', drop=True, inplace=True)
    #df_groundTrue[['Start','Length']] = df_groundTrue[['Start','Length']].astype(np.int32)
    
    #Se lee el archivo .tab generado por la red
    df_pred = pd.read_csv(path_pred, sep='\t').replace({' ':''}, regex=True)
    df_pred.columns = [i.replace(' ','').replace('|','') for i in list(df_pred.columns)]
    df_pred.set_index('id', drop=True, inplace=True)
    df_pred[['ProbabilityPresence','ProbabilityClass']] = df_pred[['ProbabilityPresence','ProbabilityClass']].astype(np.float32)
    #df_pred[['Start','Length']] = df_pred[['Start','Length']].astype(np.int32)
    #Se abre un archivo donde se guardarán las métricas
    file = open(name_file, 'w')
    
    #for th_class in threshold_class:
    #file.write(f'Analysis for the specie of InpactorDB\n')
    #file.write(f'Domain \t Threshold \t Precision \t Recall \t F1_score \n')
    
    pred_ids,ground_ids,union_ids = extract_ids(groundT=df_groundTrue, pred=df_pred)  
    print('Numero de ids unicos para el archivo de prediccion: ',len(pred_ids))
    print('Numero de ids unicos para el archivo de ground True: ',len(ground_ids))
    print('Numero de ids que se comparten en los dos archivos: ',len(union_ids))
    for th in threshold:
        print("Metricas para un threshold de ",th)
        print()
        metrics(th, df_groundTrue, df_pred, union_ids)#, pred_ids, ground_ids)
        print()
    print()
    #for th in threshold:
    #    Precision, Recall, f1, _ = metrics(th, df_groundT=df_groundTrue, df_pred=df_pred, ids=ids)
    #    file.write(f'Global \t {th} \t {Precision} \t {Recall} \t {f1} \n')

main()
"""
Precision = 0.9233617242672754 
 Recall = 0.945314965272301 
 Accuracy = 0.94087768  
 F1 = 0.9342093912720209
"""