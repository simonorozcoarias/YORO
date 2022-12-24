from utils.compareAnotation import analysis
import time 

filename = '/shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/YoloDNA_LTR/Oryza_sativa_ssp._Indica.tab'
file_csv = '/shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/YoloDNA/metrics/metrics/genomes_links.csv'
path_anotation = '/shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/YoloDNA/metrics/dataset_intact_LTR-RT'
genome = '/shared/home/sorozcoarias/coffea_genomes/Simon/YOLO/YoloDNA_LTR/R498_Chr.fasta'
idx = 188
threshold_presence = 0.85
inpactorTest = None
begin1 = time.time() 
path_pred_anot = filename
path_analysis = filename.replace('tab','metrics')
analysis(file_csv, path_anotation, idx, path_pred_anot, path_analysis, threshold = threshold_presence, inpactorTest = inpactorTest, genome=genome)
finish1 = time.time() - begin1
print("The analysis file was writeen at: ",path_analysis)
print("Analysis Executed: time elapsed: {}s".format(finish1))