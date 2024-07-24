# YORO: Pipeline for LTR-Retrotransposon domains detection and classification using Deep Neural Network YOLO inspired
## Genomic Object Detection: An Improved Approach for Transposable Element Detection and Classification Using Convolutional Neural Networks


## Installation
<a name="installation"/>

We highly recommend to use and install Python packages within an Anaconda environment. First, download the lastest version of YORO

```
git clone https://github.com/simonorozcoarias/YORO.git
```

Go to the YORO folder and find the file named "requirements.txt". Then, create and setup the environment: 

```
conda env create --name YORO --file environment.yml
conda activate YORO

```

## Testing
After successfully installing YORO, you can test it by using any genome. In this case, we will use the genome of Oryza Sativa:

```
conda activate YORO
```
Then, you must run the following command:

```
wget -O test/rice.fasta http://rice.uga.edu/pub/data/Eukaryotic_Projects/o_sativa/annotation_dbs/pseudomolecules/version_7.0/all.dir/all.con

python3 pipelineDomain.py -f test/rice.fasta -d temp
```

Finally compare your results in the path 'temp/output.tab' with the file in 'test/output.tab'. If you obtain similar (or also the same) results, congrats! the YORO pipeline is now installed and funcional.


## Parameters
To obtain all parameters for pipeline execution, please run: 
```
python3 pipelineDomain.py -h
```
All parameters that can be configurated are:

* -h or --help: show this help message and exit.
* -f FASTA_FILE or --file FASTA_FILE: Fasta file containing DNA sequences **(required)**.
* -o OUTPUTNAME or --outputName name: Filename of the output file. Default: output.tab
* -d DIRECTORY or --directory DIRECTORY: Output Path to save output file. Default: Current directory.
* -p THREADS or --threads THREADS: Number of threads to be used by YORO. Default: all available threads.
* -t THRESHOLD or --threshold THRESHOLD: Threshold value for presence filter. Default: 0.6.
* -c CYCLES or --cycles CYCLES: Numbers of cycles of detections. Default: 1
* -w WINDOW or --window WINDOW: Window size for object detection and secuence splitting. Default: 50000
* -m MODELPATH  or -modelpath MODELPATH: Path to models weights file. Default: models/ALL_DOMAINS_REDUNDANTYOLO_domain_v15.hdf5


## Fast execution
To execute this pipeline you just need provide the fasta file for predictions and run as follows:

```
python3 pipelineDomain.py -f FASTA_FILE.fasta
```
if you wish a different name for output file, run as follows:
```
python3 pipelineDomain.py -f FASTA_FILE.fasta -o ModifidedOutput.tab
```

## Pipeline Output
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

## Citation
If you use any material from this repository, please cite us as following:

Orozco-Arias, S., Lopez-Murillo, L. H., Piña, J. S., Valencia-Castrillon, E., Tabares-Soto, R., Castillo-Ossa, L., Isaza, G. & Guyot, R. (2023). Genomic object detection: An improved approach for transposable elements detection and classification using convolutional neural networks. Plos one, 18(9), e0291925.
