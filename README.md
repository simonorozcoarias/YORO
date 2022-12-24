# YOLO-DNA: Pipeline for LTR-Retrotransposones detection using Deep Neural Network YOLO inspired


## Installation:
<a name="installation"/>

We highly recommend to use and install Python packages within an Anaconda environment. First, download the lastest version of YOLO-DNA

```
git clone https://github.com/simonorozcoarias/YoloDNA.git
```

Go to the YoloDNA folder and find the file named "requirements.txt". Then, create and setup the environment: 

```
conda create -n YoloDNA python=3
conda activate YoloDNA
pip install -r requirements.txt
```

## Testing:
<a name="testing"/>

After successfully installing YoloDNA, you can test it using the testing data contained in this repository. To do so, first you must activate the conda environment:
```
conda activate YoloDNA
```
Then, you must run the following command:
```
python3 pipelineDomain.py -f test/TEx3.fasta -d temp
```

Finally compare your results in the folder temp/output.tab with the file in the folder test 'output.tab'. If you obtain similar (or also the same) results, congrats! the YoloDNA pipeline is now installed and funcional.


## Parameters
<a name="parameters"/>

To obtain all parameters for pipeline execution, please run: 
```
python3 pipelineDomain.py -h
```
All parameters that can be configurated are:

* -h or --help: show this help message and exit.
* -f FASTA_FILE or --file FASTA_FILE: Fasta file containing DNA sequences **(required)**.
* -o OUTPUTNAME or --outputName name: Filename of the output file. Default: output.tab
* -d DIRECTORY or --directory DIRECTORY: Output Path to save output file. Default: Current directory.
* -p THREADS or --threads THREADS: Number of threads to be used by YoloDNA. Default: all available threads.
* -t THRESHOLD or --threshold THRESHOLD: Threshold value for presence filter. Default: 0.6.
* -n NMS or --nms NMS: Non-Max supression value to filter secuences. Default: 0.1
* -w WINDOW or --window WINDOW: Window size for object detection and secuence splitting. Default: 50000
* -m MODELPATH  or -modelpath MODELPATH: Path to models weights file. Default: models/ALL_DOMAINS_REDUNDANTYOLO_domain_v15.hdf5


## Fast execution
<a name="fastexe"/>

To execute this pipeline you just need provide the fasta file for predictions and run as follows:

```
python3 pipelineDomain.py -f FASTA_FILE.fasta
```
if you wish a different name for output file, run as follows:
```
python3 pipelineDomain.py -f FASTA_FILE.fasta -o ModifidedOutput.tab
```

## Pipeline Output
<a name="output"/>

This tool produces one tabular file with this format:

```
|id	 |ProbabilityPresence	 |Start	 |Length	 |Class	 |ProbabilityClass

```
where:
* id: Secuence identifier from input fasta File
* ProbabilityPresence: probability of Transposable element presence 
* Start: Predicted Initial position in the secuence
* Length: Predicted lenght of Transposable element
* Class: Type of Transposable element predicted
* ProbabilityClass: Probability of this class of transposable element.


