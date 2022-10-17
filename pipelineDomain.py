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
    parser.add_option('-t', '--threshold',dest='threshold',help='Threshold for presence filter',type=float, default=0.7)
    #parser.add_argument('-c', '--cycles',dest='cycles',help='Numbers of cycles of detections',type=int, default=1)
    #parser.add_argument('-i', '--iou',dest='iou',help='Threshold for IOU filter',type=float, default=0.9)
    parser.add_option('-n', '--nms',dest='nms',help='Threshold for NMS filter',type=float, default=0.1)
    parser.add_option('-w', '--window',dest='win',help='Window size for object detection',type=int,default=50000)
    parser.add_option('-m', '--modelpath',dest='model',help='Path to models weights',type=str,default=None)    
    parser.add_option('-x', '--index',dest='index',help='Index of name genome (1-226)',type=int,default=None)    
    parser.add_option('-T', '--test',dest='inpactorDB',help='Select a test annotation file. The columns of the file match these names <id_secuenceStartLengthDomain>.',type=str,default=None) 
    parser.add_option('-D','--download',dest='download',help='download genome',type=int,default=False)
    (options,_) = parser.parse_args()
    return options
    

def main():
    begin = time.time()
    options = argumentParser()

    file = options.fastaFile
    filename = options.filename
    outputDir = options.dire
    threads = options.threads
    threshold_presence = options.threshold
    cycles = 1 #options.cycles
    #iou_threshold = options.iou
    threshold_NMS=options.nms
    total_win_len = options.win
    modelFilepath = options.model
    file_csv = 'metrics/genomes_links.csv'
    path_anotation = 'metrics/dataset_intact_LTR-RT'
    idx = options.index
    inpactorTest = options.inpactorDB
    download = options.download
    timeout = 500

    if download == False:
        if file is None:
            print("Please insert at least a file in FASTA format")
            sys.exit(1)
    else:
        path_save = '.'
        name = download2(file_csv, timeout, path_save, idx, path_anotation)
        filename = name


    
    if filename is None:
        print("No filename provided, using 'output.tab' as output filename.\n if this file exists, it will be overwrited")
        filename = 'output.tab'
    
    if outputDir is None:
        outputDir = ''
    else:
        if not os.path.exists(outputDir):
            print("This path doesn't exists, please provide a valid path.")
            sys.exit(1)
        elif outputDir[0] != '/':
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
        threshold_presence = 0.7
    
    '''if iou_threshold > 1.0 or iou_threshold < 0.0:
        print("An error ocurred in threshold value for IOU, setting this by default 0.9")
        iou_threshold = 0.9'''

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
        modelFilepath = 'models/AAYOLO_domain_v17.hdf5'

    # Cycles for detection
    slide_win = int(total_win_len / cycles)

    #total_time = []
    #inalIds_cycles = []
    #predictions_cycles = []
    #ercentages_cycles = []
    #ltr_predicted_final_cycles = []
    
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
        
        
        print("Validating nucleotides in secuences...")
        begin1 = time.time()
        check_nucleotides_master(list_seqs, threads)
        finish1 = time.time() - begin1
        print("Nucleotides in secuences checked!: time elapsed: {}s".format(finish1))

        print("Encoding secuences into oneHot encoding")
        begin1 = time.time()
        splitted_genome = create_dataset_master(list_ids, list_seqs, threads, total_win_len, outputDir)
        finish1 = time.time() - begin1
        print("Encoded!!: time elapsed: {}s".format(finish1))

        list_seqs = None  # to clean unusable variable

        """
            Domains predictions with neural networks
        """

        begin1 = time.time()
        model = loadNNArchitecture(total_win_len,modelFilepath)
        finish1 = time.time() - begin1
        print("NN Architecture loaded: time elapsed: {}s".format(finish1))

        print("Detecting Elements with DeepLearning")
        begin1 = time.time()
        Yhat_test = model.predict(splitted_genome[:,0:4,:])
        finish1 = time.time() - begin1
        print("Prediction Executed: time elapsed: {}s".format(finish1))


        begin1 = time.time()
        Yhat_pred = NMS(Yhat_test, threshold_presence, threshold_NMS)
        finish1 = time.time() - begin1
        print("Non-Max Supression exectuded: time elapsed {}s".format(finish1))
        
        begin1 = time.time()
        label_add = label_LTR(splitted_genome[:,0:4,:],Yhat_pred,threshold_presence)
        Yhat_pred[:,:,:,0:3]=Yhat_pred[:,:,:,0:3]+label_add[:,:,:,0:3]
        Yhat_pred = np.concatenate((Yhat_pred,label_add[:,:,:,0:1]),axis=3)
        finish1 = time.time() - begin1
        print("LTR detection executed: time elapsed {}s".format(finish1))

        begin1 = time.time()
        outputfile = tabGeneration(filename,Yhat_pred,list_ids,total_win_len,threshold_presence)
        finish1 = time.time() - begin1
        print("The output of this pipeline was written at: ", outputfile)
        print("File Writting time elapsed: {}s".format(finish1))

        begin1 = time.time() 
        path_pred_anot = filename
        path_analysis = filename.replace('tab','out')
        analysis(file_csv, path_anotation, idx, path_pred_anot, path_analysis, threshold = threshold_presence, inpactorTest = inpactorTest)
        finish1 = time.time() - begin1
        print("The analysis file was writeen at: ",path_analysis)
        print("Analysis Executed: time elapsed: {}s".format(finish1))

    finish = time.time() - begin
    print("Total time elapsed for pipeline execution: {}s ".format(finish))
    
if __name__ == '__main__':
    main()