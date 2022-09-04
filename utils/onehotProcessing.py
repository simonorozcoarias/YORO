import numpy as np

def fasta2one_hot(sequence, total_win_len):
    """
        This converts a fasta sequences (in nucleotides) to one-hot representation
    """

    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    rep2d = np.zeros((1, 5, total_win_len), dtype=bool)

    for nucl in sequence:
        posLang = langu.index(nucl.upper())
        rep2d[0][posLang][posNucl] = 1
        posNucl += 1
    return rep2d









def nt_TE(y,ventana,sample,threshold_presence,distancia=30):
  nucleotidos = np.zeros((sample,1,ventana))
  mask = (y[:,:,:,0:1]>=threshold_presence)*1
  valores =[]
  for i in range(y.shape[0]):
    indices = np.nonzero(mask[i,0,:,0])[0]
    for h in range(len(indices)):
      size = dicc_size[np.nonzero(y[i,0,indices[h],3:]==np.amax(y[i,0,indices[h],3:]))[0][0]]
      if h!=0 and (indices[h]-indices[h-1])<distancia:
        j1=indices[h-1]
        j2=indices[h]
        inicio = int(j1*100+y[i,0,j1,1]*100)
        inicio2=int(j2*100+y[i,0,j2,1]*100)
        fin = int(inicio2+y[i,0,j2,2]*size)
        nucleotidos[i,0,inicio:fin]=1
  return nucleotidos





y = NMS(Y_hat, threshold_presence, 0.1)
  inicio_hat_pred=[]
  inicio_true=[]
  fin_hat_pred=[]
  fin_true=[]
  indices=[]
  longitud=[]
  malos=''
  Y_hat_nt = nt_TE(y,ventana,y.shape[0],threshold_presence)
  Y_true_nt = nt_TE(Y_true,ventana,Y_hat.shape[0],threshold_presence)
  for i in range(y.shape[0]):
    #indices_hat = np.nonzero(Y_hat_nt[i,0,:])
    indices_hat_start,indices_hat_end,_ = index_pos(Y_hat_nt[i,0,:])
    indices_true_start,indices_true_end,long_true = index_pos(Y_true_nt[i,0,:])
    indices_hat_start,indices_hat_end,indices_true_start,indices_true_end,long_true=acoplamiento(indices_hat_start,indices_hat_end,indices_true_start,indices_true_end,long_true)
    array_inicio_hat = np.array(indices_hat_start)
    array_inicio_true = np.array(indices_true_start)
    error = np.absolute(array_inicio_hat-array_inicio_true)
    if np.sum((error>5000)*(error<15000)*1)>0:
      malos=malos+str(i)+'-'
    try:
      inicio_hat_pred = inicio_hat_pred + indices_hat_start
      fin_hat_pred = fin_hat_pred + indices_hat_end
    except:
      print('algo salio mal')
      pass
    
    inicio_true = inicio_true + indices_true_start
    fin_true = fin_true + indices_true_end
    longitud = longitud + long_true
    if len(inicio_hat_pred)<len(inicio_true):
      while len(inicio_hat_pred)<len(inicio_true):
        inicio_hat_pred.append(0)
        fin_hat_pred.append(0)
    if len(inicio_hat_pred)>len(inicio_true):
      while len(inicio_hat_pred)>len(inicio_true):
        inicio_true.append(inicio_true[-1])
        fin_true.append(fin_true[-1])
        indices.append(int(len(inicio_true)-1))
        longitud.append(longitud[-1])
  plt.figure()
  plt.scatter(inicio_true, inicio_hat_pred, marker='.', s=10, color='k', linewidths=1)
  #plt.plot(inicio_true,inicio_hat_pred,'bo',linewidth=0.2)
  plt.plot([0,ventana],[0,ventana],'k',linewidth=0.5)
  plt.plot([400,ventana],[0,ventana-400],'k--',linewidth=0.5)
  plt.plot([0,ventana-400],[400,ventana],'k--',linewidth=0.5)
  plt.xlim([0, ventana])
  plt.ylim([0, ventana])
  plt.xlabel('real')
  plt.ylabel('predicted')
  plt.show()

  start=np.array(inicio_true)
  start_hat=np.array(inicio_hat_pred)
  R2=1-np.sum((start-start_hat)**2)/np.sum((start-np.mean(start))**2)
  print('R^2 = '+str(R2))

  end=np.array(fin_true)
  end_hat=np.array(fin_hat_pred)
  R2=1-np.sum((end-end_hat)**2)/np.sum((end-np.mean(end))**2)
  print('R^2 = '+str(R2))












  Y_true_nt = nt_TE(y_true,ventana,y_true.shape[0],1)
  Precision = []
  Recall = []
  Sensitivity = []
  Specificity_1 = []
  producto_maximo=0
  th_vector = np.arange(0.7,1,0.01)
  for th in th_vector:
    Yhat_pred = NMS(y_hat, th, 0.1)
    Y_pred_nt = nt_TE(Yhat_pred,ventana,Yhat_pred.shape[0],th)
    TP = np.sum(Y_true_nt*Y_pred_nt)
    FP = np.sum(Y_pred_nt)-TP
    FN = np.sum(Y_true_nt)-TP
    TN = Y_true_nt.shape[0]*Y_true_nt.shape[2]-TP-FP-FN
    Precision.append(TP/(TP+FP))
    Recall.append(TP/(TP+FN))
    Sensitivity.append(TP/(TP+FN))
    Specificity_1.append(1-TN/(FP+TN))
    producto=Precision[-1]*Recall[-1]
    if producto>producto_maximo:
      th_max=th
      producto_maximo=producto