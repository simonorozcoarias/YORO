import os
import pandas as pd
import subprocess


threads = 32

data = pd.read_csv("metrics/genomes_links_final_final.csv",sep=';')

numSamples = data.shape[0]
print(f"El archivo tiene {numSamples} muestras")

index = range(0,numSamples)

for id in index:
    command = f'python -x {id} -D True -d test/ -p {threads}'
    try:
        print("#####################################################################")
        print(command)
        subprocess.run(command)
        print("#########################¡¡DONE!!####################################")
    except:
        print("Ocurrió un error para el index ",id)
    