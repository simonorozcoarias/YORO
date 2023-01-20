import pandas as pd
import numpy as np
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
  if 'Data/' in name:
    return name[5:]
  return name

@ray.remote
def ltrharvest(name):
  #subprocess.run(f'gt ltrharvest -index {name} -gff3 Data/{name}.gff3 -seqids yes -maxlenltr 3000 -similar 80', shell=True, check=True)
  subprocess.run(f'gt ltrharvest -index {name} -gff3 Data/{name}.gff3 -seqids yes -maxlenltr 7000 -mintsd 4 -maxtsd 6 -similar 85 -vic 10 -seed 20', shell=True, check=True)
  return True

def genome_generation(df_ltr, genoma, file):
  resultado = open(f'{file}_LTR_predicted.fasta','w')
  for id in df_ltr.index.unique().tolist():
      Y_pred = df_ltr.loc[id].copy()
      yy_start = np.array(Y_pred["real_start"])
      yy_end = np.array(Y_pred["real_end"])
      if yy_start.size ==1:
        yy_start = np.array([yy_start])
        yy_end = np.array([yy_end])
      for i in range(yy_start.shape[0]):
        inicio = yy_start[i]
        fin = yy_end[i]
        resultado.write(f">{id}-{inicio}-{fin}-{fin-inicio}\n")
        resultado.write(f"{genoma[id][inicio:fin]}\n")
  resultado.close()
  return None

def metrii(file, new_file, genoma, path_query):
  point1 = time.time()
  wraper =open(new_file,'r')
  fasta = list(wraper)
  wraper.close()
  num_cores = 8
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

  #subprocess.run('python pipelineDomain.py -t 0.75 -x 188 --download False -f Data/Oryza_sativa_ssp._Indica.fasta -o Data/Oryza_sativa_ssp._Indica.tab', shell=True, check=True)
  # python pipelineDomain.py -f Data/Oryza_EDTA.fasta -o Data/Oryza_EDTA.tab -t 0.75
  #subprocess.run('python pipelineDomain.py -t 0.75 -x 41 --download False -f Data/Ziziphus_jujuba.fasta -o Data/Ziziphus_jujuba.tab', shell=True, check=True)
  #subprocess.run('python pipelineDomain.py -t 0.75 -x 28 --download False -f Data/Rosa_chinensis.fasta -o Data/Rosa_chinensis.tab', shell=True, check=True)
  #subprocess.run('python pipelineDomain.py -t 0.75 -x 206 --download False -f Data/Zea_mays.fasta -o Data/Zea_mays.tab', shell=True, check=True)
  #subprocess.run('python pipelineDomain.py -t 0.75 -x 140 --download False -f Data/Coffea_canephora.fasta -o Data/Coffea_canephora.tab', shell=True, check=True)
  point2 = time.time()
  ray.init()
  lista_suf =[]
  lista_harv =[]
  for i in range(num_cores):
    lista_suf.append(suffixerator.remote(f'{new_file}{i}'))
    lista_harv.append(ltrharvest.remote(lista_suf[i]))
  z = ray.get(lista_harv)

  if 'Data/' in new_file:
    all_files = glob.glob(f'{new_file[5:]}*')
  else:
    all_files = glob.glob(f'{new_file}*')
  for path in all_files:
    os.remove(path)

  point3 = time.time()
  print(f'time is {point3-point2}')

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

  for i in range(num_cores):
    os.remove(f'{new_file}{i}')
    os.remove(f'{new_file}{i}.gff3')

  df_ltr = pd.concat(df_list, axis=0)
  df_list = None
  df_ltr = df_ltr[['id', 'motif', 'start', 'end']]
  df_ltr = df_ltr[df_ltr['motif']=='repeat_region']

  df_ltr['real_start'] = df_ltr.apply(lambda row: int(row.id.split('-')[1]) + int(row.start), axis = 1)
  df_ltr['real_end'] = df_ltr.apply(lambda row: int(row.id.split('-')[1]) + int(row.end), axis = 1)
  df_ltr['id'] = df_ltr.apply(lambda row: (row.id.split('-')[0]), axis = 1)
  df_ltr = df_ltr[['id','real_start','real_end']]
  df_ltr.sort_values(['id','real_start'],inplace = True)
  df_ltr.set_index('id',drop=True,inplace=True)
  genome_generation(df_ltr, genoma, file)
  print(df_ltr.columns)
  print(df_ltr.head())

  if path_query == None:
    return 0, 0, 0, 0

  # Aquí se modifica el gff real para solo incluir TE completos y no dominios
  df_True = pd.read_csv(path_query, sep='\t')
  df_True.columns = [i.replace(' ','') for i in list(df_True.columns)]
  df_True = df_True[['Chromosome','LTR_ID']].groupby(['LTR_ID']).first().reset_index()
  df_True['real_start'] = df_True.apply(lambda row: int(row.LTR_ID.split('_')[-2]), axis=1)
  df_True['real_end'] = df_True.apply(lambda row: int(row.LTR_ID.split('_')[-1]), axis=1)

  df_True = df_True[['Chromosome','real_start','real_end']]
  df_True.sort_values(['Chromosome','real_start'],inplace = True)
  df_True.set_index('Chromosome', inplace=True)
  print(df_True.columns)
  print(df_True.head())

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

  point4 = time.time()


  finish = time.time()
  print(f'La division del fasta se demoró {point2-point1}')
  print(f'Genome tools se demoró {point3-point2}')
  print(f'Las metricas se demoraron {point4-point3}')
  return Precision, Recall, Accuracy, F1