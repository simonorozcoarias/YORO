import numpy as np
import pandas as pd
from Bio import SeqIO
from tensorflow.keras import backend as K 

import subprocess
import time
import ray
import glob
import os

@ray.remote
def suffixerator(name):
  subprocess.run(f'gt suffixerator -db {name} -suf yes -lcp yes', shell=True, check=True)
  name = name.split('/')[-1]
  return name

@ray.remote
def ltrharvest(name):
  subprocess.run(f'gt ltrharvest -index {name} -gff3 {name}.gff3 -seqids yes -maxlenltr 3000 -similar 80', shell=True, check=True)
  #subprocess.run(f'gt ltrharvest -index {name} -gff3 Data/{name}.gff3 -seqids yes -maxlenltr 7000 -mintsd 4 -maxtsd 6 -similar 85 -vic 10 -seed 20', shell=True, check=True)
  return True

def purge(df_ltr, df_internal):
  df_ltr = df_ltr.copy()
  df_internal = df_internal.copy()
  df_internal.sort_values(['id','real_start'] , inplace = True)
  df_internal.set_index('id', inplace = True)
  df_ltr_grouped = df_ltr.groupby('id')
  count = 0
  for id, Y_after in df_ltr_grouped:
    yy_after_start = np.array(Y_after["real_start"])
    yy_after_end = np.array(Y_after["real_end"])
    if yy_after_start.size ==1:
      yy_after_start = np.array([yy_after_start])
      yy_after_end = np.array([yy_after_end])
    Y_before = df_internal.loc[id].copy()
    yy_before_start = np.array(Y_before["real_start"])
    yy_before_end = np.array(Y_before["real_end"])
    if yy_before_start.size ==1:
      yy_before_start = np.array([yy_before_start])
      yy_before_end = np.array([yy_before_end])
    k= 0
    for i in range(len(yy_after_start)):
      for j in range(k,len(yy_before_start)):
        #if ((yy_before_start[j] > yy_after_start[i]) and (yy_before_start[j] < yy_after_end[i])) or ((yy_before_end[j] > yy_after_start[i]) and (yy_before_end[j] < yy_after_end[i])):
        if (yy_before_start[j] > yy_after_start[i]) and (yy_before_end[j] < yy_after_end[i]):
          k = j
          break
        if yy_before_start[j] > yy_after_end[i]:
          try:
            df_ltr.drop(df_ltr[(df_ltr['real_start']==yy_after_start[i]) & (df_ltr['id']==id) & (df_ltr['real_end']==yy_after_end[i])].index[0], inplace = True)
            k = j
            break
          except:
            k = j
            count += 1
            break
  print(count)
  return df_ltr

def clustering(A,dist_max,dom_min):
  dom_min = dom_min-1
  B = []
  C = []
  for i in range(len(A)-1):
    B.append((A[i],A[i+1]))
  for i in range(len(A)-1):
    C.append(A[i+1][0]-A[i][1])
  D = (np.array(C)<=dist_max)*1
  E = D.copy()
  a=0
  for i in range(len(D)):
    if D[i]==1:
      a=a+1
    if (D[i]==0 or i==len(D)-1) and i!=0:
      if (i==len(D)-1) and D[i]!=0:
        inicio = i
      else:
        inicio = i-1
      for j in range(inicio,-1,-1):
        if D[j]==1:
          E[j]=a
          if j==0:
            a=0
        else:
          a=0
          break
  F = list((E>=dom_min)*1)
  H = np.array([0]+F+[0])
  J = H[1:]-H[0:-1]
  inicio = np.nonzero((J==1)*1)[0]
  fin = np.nonzero((J==-1)*1)[0]
  lista_domains =[]
  for k in range(len(inicio)):
    start = B[inicio[k]][0][0]
    end = B[fin[k]-1][1][1]
    lista_domains.append((start,end))
  return lista_domains

def metrics_internal_regions(df_ltr, genoma, path_reference):
  df_True = pd.read_csv(path_reference)
  df_True.sort_values(['sequence','real_start'] , inplace = True)
  df_True.set_index('sequence', drop=True, inplace=True)

  dist_max = 4000
  dom_min = 5
  ids_pred = df_ltr.index.unique().tolist()
  ids_ground = df_True.index.unique().tolist()
  ids_union = [value for value in ids_pred if value in ids_ground]
  ids_pred_unique = list(set(ids_pred) - set(ids_union))
  ids_ground_unique = list(set(ids_ground) - set(ids_union))
  TP = 0; FP = 0; FN = 0; TN=0
  cont = 0
  for id in ids_union:
    size = len(genoma[id])
    Y_true_start = np.array(df_True.loc[id]['real_start'])
    Y_true_end = np.array(df_True.loc[id]['real_end'])
    Y_true_nt = np.zeros((size))
    if Y_true_start.size == 1:
      Y_true_start = np.array([Y_true_start])
      Y_true_end = np.array([Y_true_end])
    for i in range(Y_true_start.shape[0]):
      Y_true_nt[Y_true_start[i]:Y_true_end[i]]=1
    Y_pred = df_ltr.loc[id].copy()
    Y_pred_start = np.array(Y_pred['Start'])
    Y_pred_end = Y_pred_start + np.array(Y_pred['Length'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    indices2 = [(Y_pred_start[i],Y_pred_end[i]) for i in range(len(Y_pred_start))]
    doms = clustering(A=indices2, dist_max=dist_max, dom_min=dom_min)
    cont = cont + len(doms)
    for tupla in doms:
      Y_pred_nt[tupla[0]:tupla[1]]=1
    TP_window = np.sum(Y_true_nt*Y_pred_nt)
    TP += TP_window
    FP += np.sum(Y_pred_nt)-TP_window
    FN += np.sum(Y_true_nt)-TP_window
    TN += size-TP_window-(np.sum(Y_pred_nt)-TP_window)-(np.sum(Y_true_nt)-TP_window)

  for id in ids_pred_unique:
    size = len(genoma[id])
    Y_pred = df_ltr.loc[id].copy()
    Y_pred_start = np.array(Y_pred['Start'])
    Y_pred_end = Y_pred_start + np.array(Y_pred['Length'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    indices2 = [(Y_pred_start[i],Y_pred_end[i]) for i in range(len(Y_pred_start))]
    doms = clustering(A=indices2, dist_max=dist_max, dom_min=dom_min)
    cont = cont + len(doms)
    for tupla in doms:
      Y_pred_nt[tupla[0]:tupla[1]]=1
    TP_window = 0
    TP += 0
    FP += np.sum(Y_pred_nt)-TP_window
    FN += 0
    TN += size-TP_window-(np.sum(Y_pred_nt)-TP_window)

  for id in ids_ground_unique:
    size = len(genoma[id])
    Y_true_start = np.array(df_True.loc[id]['real_start'])
    Y_true_end = np.array(df_True.loc[id]['real_end'])
    Y_true_nt = np.zeros((size))
    if Y_true_start.size == 1:
      Y_true_start = np.array([Y_true_start])
      Y_true_end = np.array([Y_true_end])
    for i in range(Y_true_start.shape[0]):
      Y_true_nt[Y_true_start[i]:Y_true_end[i]]=1
    TP_window = 0
    TP += TP_window
    FP += 0
    FN += np.sum(Y_true_nt)-TP_window
    TN += size-TP_window-0-(np.sum(Y_true_nt)-TP_window)
  Precision = TP/(TP+FP+K.epsilon())
  Recall = TP/(TP+FN+K.epsilon())
  Accuracy = (TP + TN)/ (TP + FN + TN + FP+K.epsilon())
  F1 = 2* Precision*Recall/(Precision + Recall+K.epsilon())
  print(f'La cantidad de regiones internas es de {cont}')
  return Precision, Recall, Accuracy, F1

def metrics_individual_domains(df_ltr, genoma, path_reference):
  df_True = pd.read_csv(path_reference)
  df_True.sort_values(['sequence','real_start'] , inplace = True)
  df_True.set_index('sequence', drop=True, inplace=True)

  dist_max = 4000
  dom_min = 5
  ids_pred = df_ltr.index.unique().tolist()
  ids_ground = df_True.index.unique().tolist()
  ids_union = [value for value in ids_pred if value in ids_ground]
  ids_pred_unique = list(set(ids_pred) - set(ids_union))
  ids_ground_unique = list(set(ids_ground) - set(ids_union))
  TP = 0; FP = 0; FN = 0; TN=0
  cont = 0
  for id in ids_union:
    size = len(genoma[id])
    Y_true_start = np.array(df_True.loc[id]['real_start'])
    Y_true_end = np.array(df_True.loc[id]['real_end'])
    Y_true_nt = np.zeros((size))
    if Y_true_start.size == 1:
      Y_true_start = np.array([Y_true_start])
      Y_true_end = np.array([Y_true_end])
    for i in range(Y_true_start.shape[0]):
      Y_true_nt[Y_true_start[i]:Y_true_end[i]]=1
    Y_pred = df_ltr.loc[id].copy()
    Y_pred_start = np.array(Y_pred['Start'])
    Y_pred_end = Y_pred_start + np.array(Y_pred['Length'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    cont = cont + len(Y_pred_start)
    for i in range(Y_pred_start.shape[0]):
      Y_pred_nt[Y_pred_start[i]:Y_pred_end[i]]=1
    TP_window = np.sum(Y_true_nt*Y_pred_nt)
    TP += TP_window
    FP += np.sum(Y_pred_nt)-TP_window
    FN += np.sum(Y_true_nt)-TP_window
    TN += size-TP_window-(np.sum(Y_pred_nt)-TP_window)-(np.sum(Y_true_nt)-TP_window)

  for id in ids_pred_unique:
    size = len(genoma[id])
    Y_pred = df_ltr.loc[id].copy()
    Y_pred_start = np.array(Y_pred['Start'])
    Y_pred_end = Y_pred_start + np.array(Y_pred['Length'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    cont = cont + len(Y_pred_start)
    for i in range(Y_pred_start.shape[0]):
      Y_pred_nt[Y_pred_start[i]:Y_pred_end[i]]=1
    TP_window = 0
    TP += 0
    FP += np.sum(Y_pred_nt)-TP_window
    FN += 0
    TN += size-TP_window-(np.sum(Y_pred_nt)-TP_window)

  for id in ids_ground_unique:
    size = len(genoma[id])
    Y_true_start = np.array(df_True.loc[id]['real_start'])
    Y_true_end = np.array(df_True.loc[id]['real_end'])
    Y_true_nt = np.zeros((size))
    if Y_true_start.size == 1:
      Y_true_start = np.array([Y_true_start])
      Y_true_end = np.array([Y_true_end])
    for i in range(Y_true_start.shape[0]):
      Y_true_nt[Y_true_start[i]:Y_true_end[i]]=1
    TP_window = 0
    TP += TP_window
    FP += 0
    FN += np.sum(Y_true_nt)-TP_window
    TN += size-TP_window-0-(np.sum(Y_true_nt)-TP_window)
  Precision = TP/(TP+FP+K.epsilon())
  Recall = TP/(TP+FN+K.epsilon())
  Accuracy = (TP + TN)/ (TP + FN + TN + FP+K.epsilon())
  F1 = 2* Precision*Recall/(Precision + Recall+K.epsilon())
  print(f'La cantidad de dominios individuales detectados es {cont}')
  return Precision, Recall, Accuracy, F1

def metrics_individual_with_ltr_harvest(df_ltr, genoma, path_reference, file, threads):
  file = file.split('/')[-1]
  new_file = file+'potential_TE'
  resultado = open(new_file,'w')
  internal_regions = open(file.split('.')[0]+'_internal_regions.csv','w')
  internal_regions.write(f"id,real_start,real_end\n")
  ext = 8000
  dist_max = 4000
  dom_min = 5
  for id in df_ltr.index.unique().tolist():
      Y_pred = df_ltr.loc[id].copy()
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
        internal_regions.write(f"{id},{tupla[0]},{tupla[1]}\n")
  resultado.close()
  internal_regions.close()

  wraper =open(new_file,'r')
  fasta = list(wraper)
  wraper.close()
  num_cores = threads
  num = len(list(fasta))
  trozo = int(num/(num_cores*2)+1)*2
  veces = int(num/trozo)
  trozo = [trozo*(i+1) for i in range(veces)]
  trozo.append(num)
  trozo = [0]+trozo
  list_fasta =[]
  for i in range(num_cores):
    list_fasta.append(fasta[trozo[i]:trozo[i+1]])
    filef = open(f'{new_file}{i}','w')
    filef.write(''.join(list_fasta[i]))
    filef.close()
  ray.init()
  lista_suf =[]
  lista_harv =[]
  for i in range(num_cores):
    lista_suf.append(suffixerator.remote(f'{new_file}{i}'))
    lista_harv.append(ltrharvest.remote(lista_suf[i]))
  z = ray.get(lista_harv)

  df_list=[]
  for i in range(num_cores):
    filef = open(f"{new_file}{i}.gff3", "r")
    texto = 'id\ttool\tmotif\tstart\tend\tpunto1\tsigno\tpunto2\tlast\n'
    for line in filef:
      if line[0]!='#':
        texto = texto + line + "\n"
    resultado = open(f'{new_file}{i}new.gff3','w')
    resultado.write(texto)
    resultado.close()
    filef.close()
    df_list.append(pd.read_csv(f'{new_file}{i}new.gff3', sep='\t'))

  df_ltr = pd.concat(df_list, axis=0, ignore_index =True)
  df_list = None
  df_ltr = df_ltr[['id', 'motif', 'start', 'end']]
  df_ltr = df_ltr[df_ltr['motif']=='repeat_region']
  df_ltr['real_start'] = df_ltr.apply(lambda row: int(row.id.split('-')[1]) + int(row.start), axis = 1)
  df_ltr['real_end'] = df_ltr.apply(lambda row: int(row.id.split('-')[1]) + int(row.end), axis = 1)
  df_ltr['id'] = df_ltr.apply(lambda row: (row.id.split('-')[0]), axis = 1)
  df_ltr = df_ltr[['id','real_start','real_end']]
  df_ltr.sort_values(['id','real_start'],inplace = True)
  df_internal = pd.read_csv(file.split('.')[0]+'_internal_regions.csv', sep=',')
  df_ltr = purge(df_ltr, df_internal)
  df_ltr.set_index('id',drop=True,inplace=True)

  df_True = pd.read_csv(path_reference)
  df_True.sort_values(['sequence','real_start'],inplace = True)
  df_True.set_index('sequence', drop=True, inplace=True)
  df_True['lon'] = df_True['real_end'] - df_True['real_start']
  #df_True[['real_start','real_end','lon']].to_csv('Oryza_true.csv',index =True)
  df_ltr.to_csv(f'{file.split(".")[0]}_pred.csv',index =True)

  # Aquí se calculan las metricas
  ids_pred = df_ltr.index.unique().tolist()
  ids_ground = df_True.index.unique().tolist()
  ids_union = [value for value in ids_pred if value in ids_ground]
  ids_pred_unique = list(set(ids_pred) - set(ids_union))
  ids_ground_unique = list(set(ids_ground) - set(ids_union))
  TP = 0; FP = 0; FN = 0; TN=0
  for id in ids_union:
    size = len(genoma[id])
    Y_true_start = np.array(df_True.loc[id]['real_start'])
    Y_true_end = np.array(df_True.loc[id]['real_end'])
    Y_true_nt = np.zeros((size))
    if Y_true_start.size == 1:
      Y_true_start = np.array([Y_true_start])
      Y_true_end = np.array([Y_true_end])
    for i in range(Y_true_start.shape[0]):
      Y_true_nt[Y_true_start[i]:Y_true_end[i]]=1
    Y_pred_start = np.array(df_ltr.loc[id]['real_start'])
    Y_pred_end = np.array(df_ltr.loc[id]['real_end'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    for i in range(Y_pred_start.shape[0]):
      Y_pred_nt[Y_pred_start[i]:Y_pred_end[i]]=1
    TP_window = np.sum(Y_true_nt*Y_pred_nt)
    TP += TP_window
    FP += np.sum(Y_pred_nt)-TP_window
    FN += np.sum(Y_true_nt)-TP_window
    TN += size-TP_window-(np.sum(Y_pred_nt)-TP_window)-(np.sum(Y_true_nt)-TP_window)

  for id in ids_pred_unique:
    size = len(genoma[id])
    Y_pred_start = np.array(df_ltr.loc[id]['real_start'])
    Y_pred_end = np.array(df_ltr.loc[id]['real_end'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    for i in range(Y_pred_start.shape[0]):
      Y_pred_nt[Y_pred_start[i]:Y_pred_end[i]]=1
    TP_window = 0
    TP += 0
    FP += np.sum(Y_pred_nt)-TP_window
    FN += 0
    TN += size-TP_window-(np.sum(Y_pred_nt)-TP_window)

  for id in ids_ground_unique:
    size = len(genoma[id])
    Y_true_start = np.array(df_True.loc[id]['real_start'])
    Y_true_end = np.array(df_True.loc[id]['real_end'])
    Y_true_nt = np.zeros((size))
    if Y_true_start.size == 1:
      Y_true_start = np.array([Y_true_start])
      Y_true_end = np.array([Y_true_end])
    for i in range(Y_true_start.shape[0]):
      Y_true_nt[Y_true_start[i]:Y_true_end[i]]=1
    TP_window = 0
    TP += TP_window
    FP += 0
    FN += np.sum(Y_true_nt)-TP_window
    TN += size-TP_window-0-(np.sum(Y_true_nt)-TP_window)
  Precision = TP/(TP+FP+K.epsilon())
  Recall = TP/(TP+FN+K.epsilon())
  Accuracy = (TP + TN)/ (TP + FN + TN + FP+K.epsilon())
  F1 = 2* Precision*Recall/(Precision + Recall+K.epsilon())
  print(f"Precision: {Precision}\nRecall: {Recall}\nAccuracy: {Accuracy}\nF1: {F1}")
  all_files = glob.glob(f"{new_file}*")
  for path in all_files:
    os.remove(path)
  return Precision, Recall, Accuracy, F1

def metrics(df_pred, file, path_reference, type_metrics, threads):
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
    if type_metrics == 'internal-regions':
      return metrics_internal_regions(df_pred, genoma, path_reference)
    elif type_metrics == 'individual-domains':
      return metrics_individual_domains(df_pred, genoma, path_reference)
    elif type_metrics == 'with-ltrharvest':
      return metrics_individual_with_ltr_harvest(df_pred, genoma, path_reference, file, threads)
    print('Enter a valid metric type')
    return 0,0,0,0

def analysis(path_pred, name_file, file_fasta, path_reference, type_metrics, threads):
    #Se lee el archivo .tab generado por la red
    df_pred = pd.read_csv(path_pred, sep='\t').replace({' ':''}, regex=True)
    df_pred.columns = [i.replace(' ','').replace('|','') for i in list(df_pred.columns)]
    df_pred.sort_values(['id','Start'] , inplace = True)
    df_pred.set_index('id', drop=True, inplace=True)
    df_pred = df_pred.loc[
        (df_pred['Class']=='GAG')|
        (df_pred['Class']=='RT')|
        (df_pred['Class']=='RNASEH')|
        (df_pred['Class']=='INT')|
        (df_pred['Class']=='ENV') |
        (df_pred['Class']=='AP')
    ]
    df_pred[['ProbabilityPresence','ProbabilityClass']] = df_pred[['ProbabilityPresence','ProbabilityClass']].astype(np.float32)
    #Se abre un archivo donde se guardarán las métricas
    file = open(name_file, 'w')
    Precision, Recall, Accuracy, F1 = metrics(df_pred,file_fasta,path_reference, type_metrics, threads)
    file.write(f"Precision: {Precision}\nRecall: {Recall}\nAccuracy: {Accuracy}\nF1: {F1}\n")
    print(f"Precision: {Precision}\nRecall: {Recall}\nAccuracy: {Accuracy}\nF1: {F1}\n")
