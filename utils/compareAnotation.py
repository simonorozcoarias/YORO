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

def renames_gff(df, original_domains):
    """
    Esta función convierte los nombres de los dominios del gff de anotación para que correspondan 
    con los nombres de los dominos del gff de predicción.

        df: Dataframe del gff.
        original_domains: Diccionario con los nomres de los dominios a convertir.

    """
    for i in original_domains.keys():
        df.replace(i,original_domains[i],inplace=True)

def metrics(threshold, df_groundT, df_pred, union_ids, pred_ids, ground_ids, longNormal, classes, dicc_size):
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
    TP = 0; FP = 0; FN = 0; TN=0
    for id in union_ids:
        Y_ground = df_groundT.loc[id].copy()
        Y_pred = df_pred.loc[id].copy()
        max_true, max_pred = Y_ground['Length(bp)'].max()+Y_ground['Start'].max(), Y_pred['Length'].max()+Y_pred['Start'].max()
        max_size = int(max(max_true, max_pred))
        Y_true = revert_gff(np.array(Y_ground["Start"]), np.array(Y_ground["Length(bp)"]), np.array(Y_ground["Domain"]), max_size, longNormal, classes)
        
        if Y_true is None:
            print("PROBLEMA")
            continue
        
        Y_pred = revert_gff(np.array(Y_pred["Start"]), np.array(Y_pred["Length"]), np.array(Y_pred["Class"]), max_size, longNormal, classes)
        Y_true_nt = nt_TE_LTR(Y_true, max_size, threshold, dicc_size)
        Y_pred_nt = nt_TE_LTR(Y_pred, max_size, threshold, dicc_size)
        TP_window = np.sum(Y_true_nt*Y_pred_nt)
        TP += TP_window
        FP += np.sum(Y_pred_nt)-TP_window
        FN += np.sum(Y_true_nt)-TP_window
        TN += Y_true_nt.shape[0]-TP_window-(np.sum(Y_pred_nt)-TP_window)-(np.sum(Y_true_nt)-TP_window)
    
    for id in pred_ids:
        Y_pred = df_pred.loc[id].copy()
        max_pred = Y_pred['Length'].max()+Y_pred['Start'].max()
        Y_pred = revert_gff(np.array(Y_pred["Start"]), np.array(Y_pred["Length"]), np.array(Y_pred["Class"]), max_pred, longNormal, classes)
        
        if Y_pred is None:
            continue

        Y_pred_nt = nt_TE_LTR(Y_pred,max_size, threshold, dicc_size)
        TP_window = 0
        TP += 0
        FP += np.sum(Y_pred_nt)-TP_window
        FN += 0
        TN += Y_pred_nt.shape[0]-TP_window-(np.sum(Y_pred_nt)-TP_window)   
    
    for id in ground_ids:
        Y_ground = df_groundT.loc[id].copy()
        max_true = Y_ground['Length(bp)'].max()+Y_ground['Start'].max()
        Y_true = revert_gff(np.array(Y_ground["Start"]), np.array(Y_ground["Length(bp)"]), np.array(Y_ground["Domain"]), max_true, longNormal, classes)
        
        if Y_true is None:
            continue
        
        Y_true_nt = nt_TE_LTR(Y_true,max_size, threshold, dicc_size)
        TP_window = 0
        TP += TP_window
        FP += 0
        FN += np.sum(Y_true_nt)-TP_window
        TN += Y_true_nt.shape[0]-TP_window-0-(np.sum(Y_true_nt)-TP_window)

    Precision = TP/(TP+FP+K.epsilon())
    Recall = TP/(TP+FN+K.epsilon())
    Accuracy = (TP + TN)/ (TP + FN + TN + FP+K.epsilon())
    F1 = 2* Precision*Recall/(Precision + Recall+K.epsilon())
    return Precision, Recall, Accuracy, F1

def analysis(file_csv, path_anotation, idx, path_pred, name_file, threshold, inpactorTest):
    dicc_size={0:1000,1:3000,2:2000,3:2000,4:1000,5:1000,6:10000}
    classes = {"RT":3,"GAG":4,"ENV":5,"INT":6,"AP":7,"RNASEH":8,"LTR":9}
    longNormal = {3:1000,4:3000,5:2000,6:2000,7:1000,8:1000,9:10000}
    original_domains = {'RH':'RNASEH','aRH':'RNASEH','RH/aRH':'RNASEH','PROT':'AP','intact_5ltr':'LTR','intact_3ltr':'LTR'}
    th =  threshold #threshold de presencia

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
        else:
            print('ERROR: no se encontro la ruta del archivo de anotacion')
            sys.exit(1)
    else: 
        specie='Inpactor_TEST'
        df_groundTrue = pd.read_csv(inpactorTest, sep='\t')
        df_groundTrue.columns = [i.replace(' ','').replace('|','') for i in list(df_groundTrue.columns)]
        df_groundTrue.set_index('id_secuence', drop=True, inplace=True)
        df_groundTrue = df_groundTrue.loc[
                (df_groundTrue['Domain']=='LTR')|
                (df_groundTrue['Domain']=='GAG')|
                (df_groundTrue['Domain']=='RT')|
                (df_groundTrue['Domain']=='RNASEH')|
                (df_groundTrue['Domain']=='INT')|
                (df_groundTrue['Domain']=='AP')
        ]
        df_groundTrue.rename(columns = {'Length':'Length(bp)'}, inplace=True)

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
    
    #Se abre un archivo donde se guardarán las métricas
    file = open(name_file, 'w')
    file.write(f'Analysis for the specie of {specie}\n\n')

    pred_ids,ground_ids,union_ids = extract_ids(groundT=df_groundTrue, pred=df_pred)  
    file.write(f'Numero de ids unicos para el archivo de prediccion: {len(pred_ids)}\n')
    file.write(f'Numero de ids unicos para el archivo de ground True: {len(ground_ids)}\n')
    file.write(f'Numero de ids que se comparten en los dos archivos: {len(union_ids)}\n\n')
    #file.write(f"\nMetricas para un threshold de {th}\n\n")
    Precision, Recall, Accuracy, F1 = metrics(th, df_groundTrue, df_pred, union_ids, pred_ids, ground_ids, longNormal, classes, dicc_size)
    file.write(f"Precision: {Precision} \nRecall: {Recall}\nAccuracy: {Accuracy}\nF1: {F1}\n")
