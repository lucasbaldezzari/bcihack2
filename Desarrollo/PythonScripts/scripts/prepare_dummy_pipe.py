import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

# from SignalProcessor import Filter, FeatureExtractor, Classifier
from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.CSPMulticlass import CSPMulticlass
from SignalProcessor.RavelTransformer import RavelTransformer

"""Usaremos el registro de la Syntehtic board para entrenar y usar el pipeline"""

file = "data/eegForDummyTests/eegdata/sesion1/sesion_1.0.npy"
eventosFile = "data/eegForDummyTests/eegdata/sesion1/sesion_1.0_events.txt"
cspsFolder = "data/eegForDummyTests/csps/"
classifiersFolder = "data/eegForDummyTests/classifiers/"
pipelinesFolder = "data/eegForDummyTests/pipelines/"

#Cargamos archivo
raw_eeg = np.load(file)
channels = [1,2,3,4,5,6,7,8] #canales de interés
raw_eeg = raw_eeg[channels,:]
print("raw_eeg shape:", raw_eeg.shape)

#cargamos archivo txt de eventos y lo pasamos a un dataframe
eventos = pd.read_csv(eventosFile, sep=",")
#seteamos la columna trialNumber como índice
eventos = eventos.set_index("trialNumber")
eventos

#debemos dividir la señal de EEG en trials. Cada trial es la suma del startingTime y el cueDuration 
#Nos interesa quedarnos con el cueDuration. 
#Utilizamos la frecuencia de muestreo para calcular la cantidad de muestras que representa el cueDuration

#calculamos la cantidad de muestras que representa el cueDuration
sample_rate = 250.




