#!/usr/bin/python3
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

def find(name, path):
    for root, dirs, files in os.walk(path, topdown=True):
        if name in files:
            return os.path.join(root, name)

def extract_dom(domain, groundT, pred):
    domains_groundT = ['GAG','RT','RH','aRH','RH/aRH','INT','PROT','intact_5ltr','intact_3ltr']
    domains_pred = ['GAG','RT','RNASEH','INT','AP','LTR']
    pos_groundT = []
    pos_pred = []
    if domain==domains_pred[0]:
        pos_groundT = groundT.loc[groundT['Domain']==domains_groundT[0]]
        pos_pred = pred.loc[pred['Class']==domains_pred[0]]
    elif domain==domains_pred[1]:
        pos_groundT = groundT.loc[groundT['Domain']==domains_groundT[1]]
        pos_pred = pred.loc[pred['Class']==domains_pred[1]]
    elif domain==domains_pred[2]:
        pos_groundT = groundT.loc[(groundT['Domain']==domains_groundT[2]) | (groundT['Domain']==domains_groundT[3]) | (groundT['Domain']==domains_groundT[4])]
        pos_pred = pred.loc[pred['Class']==domains_pred[2]]
    elif domain==domains_pred[3]:
        pos_groundT = groundT.loc[groundT['Domain']==domains_groundT[5]]
        pos_pred = pred.loc[pred['Class']==domains_pred[3]]
    elif domain==domains_pred[4]:
        pos_groundT = groundT.loc[groundT['Domain']==domains_groundT[6]]
        pos_pred = pred.loc[pred['Class']==domains_pred[4]]
    elif domain==domains_pred[5]:
        pos_groundT = groundT.loc[(groundT['Domain']==domains_groundT[7]) | (groundT['Domain']==domains_groundT[8])]
        pos_pred = pred.loc[pred['Class']==domains_pred[5]]
    return pos_groundT, pos_pred


def extract_ids(groundT, pred):
    if groundT.empty==True and pred.empty==True:
        ids = []
    elif groundT.empty==False:
        ids = pred.index.unique().tolist()
    elif pred.empty==False:
        ids = groundT.index.unique().tolist()
    else:
        ids = groundT.index.unique().tolist()
        ids.append(pred.index.unique().tolist())
    return ids

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

def metrics(threshold, df_groundT, df_pred, ids):
    pred_th = df_pred.loc[df_pred['ProbabilityPresence']>=threshold].copy()
    TP = 0; FP = 0; FN = 0
    for id in ids:
        #print('ID: ',id)
        true_start, true_end = positions(df_groundT['Start'], df_groundT['Length(bp)'],id)
        pred_start, pred_end = positions(pred_th['Start'], pred_th['Length'],id)
        if true_start is None and pred_end is not None:
            l_start = np.min(pred_start)
            l_end = np.max(pred_end)
            true_vect = nucleotide(l_start, l_end)
            pred_vect = nucleotide(l_start, l_end, pred_start, pred_end)
        elif pred_start is None and true_start is not None:
            l_start = np.min(true_start)
            l_end = np.max(true_end)
            true_vect = nucleotide(l_start, l_end, true_start, true_end)
            pred_vect = nucleotide(l_start, l_end)
        elif true_start is not None and pred_start is not None:
            #print(f'Start min: {true_start} fin min: {true_end}')
            #print('Start:', [np.min(true_start),np.min(pred_start)])
            l_start = np.min([np.min(true_start),np.min(pred_start)])
            l_end = np.max([np.max(true_end),np.max(pred_end)])
            true_vect = nucleotide(l_start, l_end, true_start, true_end)
            pred_vect = nucleotide(l_start, l_end, pred_start, pred_end)
        else:
            #print('Error: no se logro los dominios para el id: ',id)
            #print('Salida: ',df_groundT.loc[id])
            continue
        #print(end-init) #Longitud de la ventana
        TP_window = np.sum(true_vect*pred_vect)
        TP += TP_window
        FP += np.sum(pred_vect)-TP_window
        FN += np.sum(true_vect)-TP_window
    list_scores = [TP,FP,FN]
    Precision = TP/(TP+FP+K.epsilon())
    Recall = TP/(TP+FN+K.epsilon())
    f1 = 2*Precision*Recall/(Precision+Recall+K.epsilon())
    return round(Precision,3), round(Recall,3), round(f1,3), list_scores

def analysis(file_csv, path_anotation, idx, path_pred, domain, name_file):
    begin = time.time()
    #Se lee el archivo csv con los genomas
    df_genome = pd.read_csv(file_csv, sep=';', skiprows=1, skipfooter=558-301, engine='python')
    specie = df_genome['Species'].loc[idx-1]
    #Se determina cual es el genoma que se desea estudiar a partir del indice en el csv
    query = str(specie.replace(' ','_')+'.txt')
    path_query = find(query, path_anotation)

    if path_query is not None:
        df_groundTrue = pd.read_csv(path_query, sep='\t')
        df_groundTrue.columns = [i.replace(' ','') for i in list(df_groundTrue.columns)]
        df_groundTrue.set_index(df_groundTrue['Chromosome'], drop=True, inplace=True)
    else:
        print('ERROR: no se encontro la ruta del archivo de anotacion')
    
    #Se lee el archivo .tab generado por la red
    df_pred = pd.read_csv(path_pred, sep='\t').replace({' ':''}, regex=True)
    df_pred.columns = [i.replace(' ','').replace('|','') for i in list(df_pred.columns)]
    df_pred.set_index(df_pred['id'], drop=True, inplace=True)
    df_pred[['ProbabilityPresence','ProbabilityClass']] = df_pred[['ProbabilityPresence','ProbabilityClass']].astype(np.float32)
    
    #El archivo se limita a aquellas anotaciones que superaron una probabilidad de clase superior al threshold
    df_pred = df_pred.loc[df_pred['ProbabilityClass']>th_class]
    file = open(name_file, 'w')
    file.write(f'Analysis for the species of {specie} \n')
    file.write(f'Domain \t Threshold \t Precision \t Recall \t F1_score \n')
    scores = {th:np.array([0,0,0]) for th in threshold}
    scores['type'] = domain.upper() 
    if domain.upper() != 'ALL':
        df_groundT, df_predict = extract_dom(domain=domain, groundT=df_groundTrue.copy(), pred=df_pred.copy())
        ids = extract_ids(groundT=df_groundT, pred=df_predict) 
        for th in threshold:
            Precision, Recall, f1, list_scores = metrics(th, df_groundT=df_groundT, df_pred=df_predict, ids=ids)
            scores[th] = scores[th] + list_scores
            file.write(f'{domain} \t {th} \t {Precision} \t {Recall} \t {f1} \n')
            print(f'Metricas con un threshold de: {th} \n Precision: {Precision} \n Recall: {Recall} \n f1: {f1} \n')
    else:
        domains = ['GAG','RT','RNASEH','INT','AP','LTR']
        for dom in domains:
            df_groundT, df_predict = extract_dom(domain=dom, groundT=df_groundTrue.copy(), pred=df_pred.copy())
            ids = extract_ids(groundT=df_groundT, pred=df_predict)  
            print('Dominio: ',dom) #,'df_groundT: ',df_groundT, 'Prediccion',df_predict)
            for th in threshold:
                #file.write(f'Thershold {th} \n')
                Precision, Recall, f1, list_scores = metrics(th, df_groundT=df_groundT, df_pred=df_predict, ids=ids)
                scores[th] = scores[th] + list_scores
                file.write(f'{dom} \t {th} \t {Precision} \t {Recall} \t {f1} \n')
                print(f'Metricas con un threshold de {th} para el dominio {dom} \n Precision: {Precision} \n Recall: {Recall} \n f1: {f1} \n')
        for th in threshold:
            precisionGlobal = scores[th][0]/(scores[th][0] + scores[th][1]+ K.epsilon())
            recallGlobal = scores[th][0]/(scores[th][0] + scores[th][2] + K.epsilon())
            f1Global = 2*precisionGlobal*recallGlobal/(precisionGlobal + recallGlobal + K.epsilon())
            file.write(f'GLOBAL \t {th} \t {precisionGlobal} \t {recallGlobal} \t {f1Global} \n')
    return scores