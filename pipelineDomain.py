#!/usr/bin/python3
'''
    This pipeline is for processing and detection of domains with a neural network architecture
    @author: Johan Sebastian Piña Duran
'''
import sys
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from optparse import OptionParser
import numpy as np
import tensorflow as tf
import multiprocessing

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.fastaProcessing import get_final_dataset_size
from utils.fastaProcessing import check_nucleotides_master
from utils.fastaProcessing import create_dataset_master

from utils.deepLutils import loadNNArchitecture
from utils.deepLutils import NMS, label_LTR

from utils.resultsWriting import tabGeneration

from utils.compareAnotation import analysis
from utils.Web_genome import download2
from utils.complete_gff import main

from utils.track_memory import track_memory_use, plot_memory_use

headerFile = f'filename,fastasize,DownloadTime,FileSplittingTime,OneHotEncodigTime,LoadingNNTime,PredictionTime,NMSTime,LabelingLTRTime,FileWritingTime,overallTime,PipelineTimeExec,maxMemConsum\n'
timesVect = []

def argumentParser():

    """
        Let's parse the pipeline parameters
    """ 

    print("\n#########################################################################")
    print("#                                                                      #")
    print("# Pipeline: A software based on deep learning to identify and classify #")
    print("#               LTR-retrotransposons domains in plant genomes          #")
    print("#                                                                      #")
    print("#########################################################################\n")

    usage = "Usage: pipelineDomain.py -f file.fasta -[options] [Value]"
    parser = OptionParser(usage=usage)

    parser.add_option('-f', '--file', dest='fastaFile', help='Fasta file containing DNA sequences. Required*',type=str, default=None)   
    parser.add_option('-o', '--outputName', dest='filename', help='Output file name.',type=str, default=None)  
    parser.add_option('-d', '--directory', dest='dire', help='Output directory',type=str, default=None)  
    parser.add_option('-p', '--threads', dest='threads', help='Number of threads to execute this pipeline',type=int, default=None)
    parser.add_option('-t', '--threshold',dest='threshold',help='Threshold for presence filter',type=float, default=0.8)
    parser.add_option('-c', '--cycles',dest='cycles',help='Numbers of cycles of detections',type=int, default=1)
    parser.add_option('-w', '--window',dest='win',help='Window size for object detection',type=int,default=50000)
    parser.add_option('-m', '--modelpath',dest='model',help='Path to models weights',type=str,default=None)
    parser.add_option('-M', '--type_metrics',dest='type_metrics',help='',type=str,default=None)    
    #Eliminar lo del csv
    #parser.add_option('-x', '--index',dest='index',help='Index of name genome (1-226)',type=int,default=None)    
    #Para métricas
    parser.add_option('-T', '--test',dest='reference_annotation',help='Select a test annotation file. The columns of the file match these names <id_secuence\tStart\tLength\tDomain>.',type=str,default=None) 
    (options,_) = parser.parse_args()
    return options
    
@track_memory_use(close=False, return_history=True,plot=False)
def main():
    begin = time.time()
    options = argumentParser()

    file = options.fastaFile
    filename = options.filename
    outputDir = options.dire
    threads = options.threads
    threshold_presence = options.threshold
    cycles = options.cycles
    total_win_len = options.win
    modelFilepath = options.model
    type_metrics = options.type_metrics

    if file is None:
        print("Please insert at least a file in FASTA format")
        sys.exit(1)

    time0 = time.time()
    timesVect.append(str(file))
    fastaSize = os.path.getsize(file)
    timesVect.append(str(fastaSize))
    #print("El tamaño del archivo es de: ",fastaSize)
    timesVect.append(str(finish1))
    
    if filename is None:
        filename = 'output.tab'
        print("No filename provided, using 'output.tab' as output filename.\n if this file exists, it will be overwrited")
    
    if outputDir is None:
        outputDir = ''
    else:
        if not os.path.exists(outputDir):
            print("This path doesn't exists, please provide a valid path.")
            sys.exit(1)
        elif outputDir[-1] != '/':
            outputDir = outputDir+'/'


    filename = outputDir+filename
    print("Output file is set to: ",filename)

    if threads is None:
        threads = multiprocessing.cpu_count()
        print("In this execution will be used {} cores".format(threads))
    elif threads > multiprocessing.cpu_count():
        threads = multiprocessing.cpu_count()
        print("Number of cores exced available resources, setting to {} cores".format(threads))
        
    if threshold_presence > 1.0 or threshold_presence < 0.0:
        print("An error ocurred in threshold value for presence, setting this by default 0.7")
        threshold_presence = 0.8

    if threshold_NMS > 1.0 or threshold_NMS < 0.0:
        print("An error ocurred in threshold value for NMS, setting this by default 0.1")
        threshold_NMS = 0.1


    if total_win_len is not None:
        print("You set a different value for the input of the Neural network pre-trained, this pipeline should fail. \n Please retrain the neural network!!")
        total_win_len = int(total_win_len)
    
    if modelFilepath is not None:
        if os.path.exists(modelFilepath):
            print("Path to weightsPath: ",modelFilepath)
            print("A different weights file was set!")
        else:
            print("An exception ocurred, the weigth file does not exist, please set a correct filenaname!")
            sys.exit(1)
    else:
        modelFilepath = 'models/AAqqYOLOqqdomainqqV25.hdf5'

    # Cycles for detection
    slide_win = int(total_win_len / cycles)
    
    for cycle in range(0,cycles):
        print("Cycle #{} from {}".format(cycle+1,cycles))
        slide = slide_win * cycle
        tf.keras.backend.clear_session()

        """
            Let's process the fasta file
        """

        begin1 = time.time()
        list_ids, list_seqs = get_final_dataset_size(file, total_win_len, slide)
        finish1 = time.time() - begin1
        print("Splited fasta File in secuences of {} nucleotides: time elapsed: {}s ".format(total_win_len,finish1))
        timesVect.append(str(finish1))

        print("Encoding secuences into oneHot encoding")
        begin1 = time.time()
        splitted_genome = create_dataset_master(list_ids, list_seqs, threads, total_win_len, outputDir)
        finish1 = time.time() - begin1
        print("Encoded!!: time elapsed: {}s".format(finish1))
        timesVect.append(str(finish1))
        list_seqs = None  # to clean unusable variable

        """
            Domains predictions with neural networks
        """

        begin1 = time.time()
        model = loadNNArchitecture(total_win_len,modelFilepath)
        finish1 = time.time() - begin1
        print("NN Architecture loaded: time elapsed: {}s".format(finish1))
        timesVect.append(str(finish1))

        print("Detecting Elements with DeepLearning")
        begin1 = time.time()
        Yhat_test = model.predict(splitted_genome[:,0:4,:])
        finish1 = time.time() - begin1
        print("Prediction Executed: time elapsed: {}s".format(finish1))
        timesVect.append(str(finish1))

        begin1 = time.time()
        Yhat_pred = NMS(Yhat_test, threshold_presence, threshold_NMS)
        finish1 = time.time() - begin1
        print("Non-Max Supression exectuded: time elapsed {}s".format(finish1))
        timesVect.append(str(finish1))

        num_domains = Yhat_pred.shape
        print("The number of detected domains are",num_domains)
        begin1 = time.time()
        outputfile = tabGeneration(filename,Yhat_pred,list_ids,total_win_len,threshold_presence)
        finish1 = time.time() - begin1
        print("The output of this pipeline was written at: ", outputfile)
        print("File Writting time elapsed: {}s".format(finish1))
        timesVect.append(str(finish1))
        
        

    finish = time.time() - begin
    finish0 = time.time() - time0
    print("Total time elapsed for pipeline execution: {}s ".format(finish))
    timesVect.append(str(finish))
    timesVect.append(str(finish0))
    resultfilecontent = headerFile + ','.join(timesVect)
    return [file,resultfilecontent]
    

if __name__ == '__main__':
    
    resultToWrite, historyMem = main()
    document = open(f"test/testResults_{resultToWrite[0].split('.')[0]}.csv",'w')
    document.write(resultToWrite[1]+','+str(max(historyMem)))
    document.close()
    print("ExecutionComplete!!!!")
