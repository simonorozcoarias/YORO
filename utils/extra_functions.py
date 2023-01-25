import numpy as np
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

def nt_TE_inside(y,ventana,threshold_presence, dicc_size):
    nucleotidos = np.zeros((ventana))
    indices = np.nonzero(y[:,0])[0]
    for h in range(0,len(indices)):
      size = dicc_size[np.argmax(y[indices[h],3:])]
      if h!=0 and (indices[h]-indices[h-1])<30 and y[indices[h],9]==0 and y[indices[h-1],9]==0:
        j1=indices[h-1]
        j2=indices[h]
        inicio = int(j1*100+y[j1,1]*100)
        inicio2=int(j2*100+y[j2,1]*100)
        fin = int(inicio2+y[j2,2]*size)
        nucleotidos[inicio:fin]=1
    return nucleotidos

def nt_TE_domain(y,ventana,threshold_presence, dicc_size):
    nucleotidos = np.zeros((ventana))
    indices = np.nonzero(y[:,0])[0]
    for h in range(0,len(indices)):
      size = dicc_size[np.argmax(y[indices[h],3:])]
      if y[indices[h],9]==0:
        j1=indices[h]
        inicio = int(j1*100+y[j1,1]*100)
        fin = int(inicio+y[j1,2]*size)
        nucleotidos[inicio:fin]=1
    return nucleotidos

def nt_TE_filter(y,ventana,threshold_presence, dicc_size):
  nucleotidos = np.zeros((ventana))
  indices = np.nonzero(y[:,0])[0]
  mask = (y[indices,9]==0)
  indices = indices[mask]
  indices2=[]
  for h in range(len(indices)):
    size = dicc_size[np.argmax(y[indices[h],3:])]
    j=indices[h]
    inicio = int(j*100+y[j,1]*100)
    fin = int(inicio+y[j,2]*size)
    indices2.append((inicio,fin))
    
  doms = clustering(A=indices2, dist_max=4000, dom_min=4)
  for tupla in doms:
    nucleotidos[tupla[0]:tupla[1]]=1
  return nucleotidos, doms
