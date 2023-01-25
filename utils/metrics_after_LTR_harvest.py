import pandas as pd
import numpy as np
from Bio import SeqIO
from tensorflow.keras import backend as K
from utils.extra_functions import clustering

def per_domain(df_ltr, df_True, genoma):
  dist_max = 4000
  dom_min = 4
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
    Y_pred = df_ltr.loc[id].copy()
    Y_pred_start = np.array(Y_pred['Start'])
    Y_pred_end = Y_pred_start + np.array(Y_pred['Length'])
    Y_pred_nt = np.zeros((size))
    if Y_pred_start.size == 1:
      Y_pred_start = np.array([Y_pred_start])
      Y_pred_end = np.array([Y_pred_end])
    indices2 = [(Y_pred_start[i],Y_pred_end[i]) for i in range(len(Y_pred_start))]
    doms = clustering(A=indices2, dist_max=dist_max, dom_min=dom_min)
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
  return Precision, Recall, Accuracy, F1


def metrii(df_ltr, df_True, genoma, path_query):
  if path_query == None:
    return 0, 0, 0, 0

  # Aquí se modifica el gff real para solo incluir dominios internos
  df_True.reset_index(inplace = True)
  print(df_True.head())
  df_True = df_True.groupby('LTR_ID').agg({'Start': 'first', 'End': 'last', 'Chromosome': 'first'})
  df_True.set_index('Chromosome', drop=True, inplace=True)
  print(df_True.head())
  df_True['real_start'] = df_True['Start']
  df_True['real_end'] = df_True['End']

  # Aquí se calculan las metricas
  Precision, Recall, Accuracy, F1 = per_domain(df_ltr, df_True, genoma)
  return Precision, Recall, Accuracy, F1