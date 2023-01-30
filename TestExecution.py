import os
import subprocess


threads = 32

index = range(0,41)

for id in index:
    command = f'python -x {id} -D True -d test/ -p {threads}'
    try:
        print("#####################################################################")
        subprocess.run(command)
        print("#########################¡¡DONE!!####################################")
    except:
        print("Ocurrió un error para el index ",id)
    