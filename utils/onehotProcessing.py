import numpy as np

def fasta2one_hot(sequence, total_win_len):
    """
        This converts a fasta sequences (in nucleotides) to one-hot representation

        this was modificated with an exception to handle the no A,C,G,T,N letter and 
        transforme it to N
    """

    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    rep2d = np.zeros((1, 5, total_win_len), dtype=bool)

    for nucl in sequence:
        try:
            posLang = langu.index(nucl.upper())
        except:
            posLang = 4

        rep2d[0][posLang][posNucl] = 1
        posNucl += 1
    return rep2d