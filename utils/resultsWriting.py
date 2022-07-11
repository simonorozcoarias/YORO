import numpy as np

def tabGeneration(resfile,Yhat_pred,list_ids,window,threshold_presence):
    '''
    This function writes the tabular file for the output

    '''
    classes = {3:"RT",4:"GAG",5:"ENV",6:"INT",7:"AP",8:"RNASEH",9:"LTR"}
    print("Writting output file!!")
    resultado = open(resfile,'w')
    resultado.write("|id\t |ProbabilityPresence\t |Start\t |Length\t |Class\t |ProbabilityClass\n")
    
    ## Acá se empieza a generar el archivo para procesar la salida.
    #print("El tamaño que se genera es: ",np.shape(Yhat_pred[0,0,:,0]) )
    for idx,_ in enumerate(Yhat_pred):
        for hundred in range(len(Yhat_pred[0,0,:,0])):
            if Yhat_pred[idx,0,hundred,0] > threshold_presence:
                #print(list_ids[idx])#.split("#")[0])## Secuencia donde se ubica
                #print(Yhat_pred[idx,0,hundred,0]) # probabilidad de presencia > 0.6
                #print(Yhat_pred[idx,0,hundred,1]*(hundred+1) *100) # posicion dentro de la division de la secuencia no el global
                #print(Yhat_pred[idx,0,hundred,1]) # posicion dentro de la division de la secuencia no el global
                inicioSeq = int(list_ids[idx].split("#")[1])
                localIni = inicioSeq + round(Yhat_pred[idx,0,hundred,1]*100*(hundred+1))
                #print("Está ubicado en la pos {} de la secuencia".format(localIni))
                longitud = Yhat_pred[idx,0,hundred,2]*window
                #print("la longitud del elemento es: {}".format(longitud))
                clasNum = np.argmax(Yhat_pred[idx,0,hundred,3:])+3
                #print("la mejor clase es: ",classes[clasNum])
                resultado.write(f"{list_ids[idx].split('#')[0]}\t {Yhat_pred[idx,0,hundred,0]}\t {localIni}\t {longitud}\t {classes[clasNum]}\t {np.amax(Yhat_pred[idx,0,hundred,3:])}\n")
    resultado.close()
    print("File writted sucessfully!!")

    return resfile