import pandas as pd
import numpy as np
from Bio import SeqIO
from tensorflow.keras import backend as K

def per_domain(df_ltr, df_True, genoma, domain):
  if domain == 'all':
    pass
  else:
    df_True = df_True[df_True['Domain']==domain]
    df_ltr = df_ltr[df_ltr['Class']==domain]
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
  return Precision, Recall, Accuracy, F1


def metrii(df_ltr, df_True, genoma, path_query):
  df_ltr['real_start'] = df_ltr['Start']
  df_ltr['real_end'] = df_ltr['Start']+df_ltr['Length']
  print(df_ltr.columns)
  print(df_ltr.head())

  if path_query == None:
    return 0, 0, 0, 0

  # Aquí se modifica el gff real para solo incluir dominios internos
  df_True['real_start'] = df_True['Start']
  df_True['real_end'] = df_True['Start']+df_True['Length(bp)']

  df_True['lon']=df_True['real_end']-df_True['real_start']
  df_ltr['lon']=df_ltr['real_end']-df_ltr['real_start']

  # Aquí se calculan las metricas
  Precision_all, Recall_all, Accuracy_all, F1_all = per_domain(df_ltr, df_True, genoma, 'all')
  Precision_GAG, Recall_GAG, Accuracy_GAG, F1_GAG = per_domain(df_ltr, df_True, genoma, 'GAG')
  Precision_RT, Recall_RT, Accuracy_RT, F1_RT = per_domain(df_ltr, df_True, genoma, 'RT')
  Precision_RNASEH, Recall_RNASEH, Accuracy_RNASEH, F1_RNASEH = per_domain(df_ltr, df_True, genoma, 'RNASEH')
  Precision_INT, Recall_INT, Accuracy_INT, F1_INT = per_domain(df_ltr, df_True, genoma, 'INT')
  Precision_AP, Recall_AP, Accuracy_AP, F1_AP = per_domain(df_ltr, df_True, genoma, 'AP')
  Precision = [Precision_all, Precision_GAG, Precision_RT, Precision_RNASEH, Precision_INT, Precision_AP]
  Recall = [Recall_all, Recall_GAG, Recall_RT, Recall_RNASEH, Recall_INT, Recall_AP]
  Accuracy = [Accuracy_all, Accuracy_GAG, Accuracy_RT, Accuracy_RNASEH, Accuracy_INT, Accuracy_AP]
  F1 = [F1_all, F1_GAG, F1_RT, F1_RNASEH, F1_INT, F1_AP]
  return Precision, Recall, Accuracy, F1