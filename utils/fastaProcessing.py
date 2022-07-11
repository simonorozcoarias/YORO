import sys
import os
import numpy as np
from Bio import SeqIO
import multiprocessing

from utils.onehotProcessing import fasta2one_hot


def get_final_dataset_size(file, total_win_len, slide):
    """
        These functions are used to calcute the size of the data to be analyze by the software
    """

    seqfile = [x for x in SeqIO.parse(file, 'fasta')]
    list_ids_splitted = []
    list_seq_splitter = []
    for i in range(len(seqfile)):
        for j in range(slide, len(str(seqfile[i].seq)), total_win_len):
            if "#" in str(seqfile[i].id):
                print("FATAL ERROR: Sequence ID (" + str(seqfile[i].id) + ") must no contain character '#', please remove "
                                     "all of these and re-run the script")
                sys.exit(0)
            initial_pos = j
            end_pos = initial_pos + total_win_len
            if end_pos > len(str(seqfile[i].seq)):
                end_pos = len(str(seqfile[i].seq))
            list_ids_splitted.append(str(seqfile[i].id) + "#" + str(initial_pos) + "#" + str(end_pos))
            list_seq_splitter.append(str(seqfile[i].seq)[initial_pos:end_pos])
    return list_ids_splitted, list_seq_splitter

def check_nucleotides_master(list_seqs, threads):
    """
        These functions are used to check if the input sequences contain non-nucleotic characters (others than A, C, T, G, N)
    """
    n = len(list_seqs)
    seqs_per_procs = int(n / threads)
    remain = n % threads
    ini_per_thread = []
    end_per_thread = []
    for p in range(threads):
        if p < remain:
            init = p * (seqs_per_procs + 1)
            end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
        else:
            init = p * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        ini_per_thread.append(init)
        end_per_thread.append(end)

    # Run in parallel the checking
    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(check_nucleotides_slave,
                                     args=[list_seqs[ini_per_thread[x]:end_per_thread[x]]]) for x in range(threads)]
    localChecks = [p.get() for p in localresults]
    for i in range(len(localChecks)):
        if localChecks[i] == 1:
            print("FATAL ERROR: DNA sequences must contain only A, C, G, T, or N characters, please fix it and "
                  "re-run Inpactor2")
            sys.exit(0)

    pool.close()
def check_nucleotides_slave(list_seqs):

    for seq in list_seqs:
        noDNAlanguage = [nucl for nucl in str(seq) if nucl.upper() not in ['A', 'C', 'T', 'G', 'N', '\n']]
        if len(noDNAlanguage) > 0:
            return 1
    return 0

def create_dataset_master(list_ids, list_seqs, threads, total_win_len, outputDir):
    """
        These functions split the input sequences into total_win_len length to get a standard size of all sequences
        needed to execute the neural networks
    """
    if outputDir == '':
        os.makedirs('temp',exist_ok=True)
        outputDir = 'temp'

    n = len(list_ids)
    seqs_per_procs = int(n / threads)
    remain = n % threads
    ini_per_thread = []
    end_per_thread = []
    for p in range(threads):
        if p < remain:
            init = p * (seqs_per_procs + 1)
            end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
        else:
            init = p * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        ini_per_thread.append(init)
        end_per_thread.append(end)
    pool = multiprocessing.Pool(processes=threads)

    localresults = [pool.apply_async(create_dataset_slave,
                                     args=[list_seqs[ini_per_thread[x]:end_per_thread[x]], total_win_len, outputDir,
                                           x]) for x in
                    range(threads)]
    localTables = [p.get() for p in localresults]

    splitted_genome = np.zeros((n, 5, total_win_len), dtype=bool)
    index = 0
    for i in range(len(localTables)):
        if localTables[i].shape[0] > 1:
            try:
                dataset = np.load(outputDir + '/dataset_2d_' + str(i) + '.npy')
                for j in range(dataset.shape[0]):
                    splitted_genome[index, :, :] = dataset[j, :, :]
                    index += 1
                os.remove(outputDir + '/dataset_2d_' + str(i) + '.npy')
            except FileNotFoundError:
                print('WARNING: I could not find: ' + outputDir + '/dataset_2d_' + str(i) + '.npy')
    pool.close()
    return splitted_genome
    
def create_dataset_slave(list_seqs, total_win_len, outputdir, x):
    j = 0
    if len(list_seqs) > 0:
        dataset = np.zeros((len(list_seqs), 5, total_win_len), dtype=bool)
        for i in range(len(list_seqs)):
            dataset[j, :, :] = fasta2one_hot(list_seqs[i], total_win_len)
            j += 1

        if dataset.shape[1] > 1:
            np.save(outputdir + '/dataset_2d_' + str(x) + '.npy', dataset.astype(np.uint8))
            return np.zeros((10, 10), dtype=bool)
        else:  # Process did not find any LTR-RT
            return np.zeros((1, 1), dtype=bool)
    else:
        # there is no elements for processing in this thread
        return np.zeros((1, 1), dtype=bool)

def get_final_dataset_size(file, total_win_len, slide):
    seqfile = [x for x in SeqIO.parse(file, 'fasta')]
    list_ids_splitted = []
    list_seq_splitter = []
    for i in range(len(seqfile)):
        for j in range(slide, len(str(seqfile[i].seq)), total_win_len):
            if "#" in str(seqfile[i].id):
                print("FATAL ERROR: Sequence ID (" + str(seqfile[i].id) + ") must no contain character '#', please remove "
                                     "all of these and re-run Inpactor2")
                sys.exit(0)
            initial_pos = j
            end_pos = initial_pos + total_win_len
            if end_pos > len(str(seqfile[i].seq)):
                end_pos = len(str(seqfile[i].seq))
            list_ids_splitted.append(str(seqfile[i].id) + "#" + str(initial_pos) + "#" + str(end_pos))
            list_seq_splitter.append(str(seqfile[i].seq)[initial_pos:end_pos])
    return list_ids_splitted, list_seq_splitter


