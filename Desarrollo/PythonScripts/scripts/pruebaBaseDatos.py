"""Script para generar un pipeline para evaluar diferentes combinaciones de hiperparámetros
y obtener el mejor modelo para clasificar las señales de EGG.
"""

import numpy as np
import pandas as pd

from TrialsHandler.TrialsHandler import TrialsHandler
from TrialsHandler.Concatenate import Concatenate

from SignalProcessor.Filter import Filter
from SignalProcessor.CSPMulticlass import CSPMulticlass
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.RavelTransformer import RavelTransformer

## Clasificadores LDA y SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

## Librerias para entrenar y evaluar el modelo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import pickle


### ********** Cargamos los datos **********
file = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_1 = np.load(file)
eventos_1 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct1_r2.npy"
eventosFile = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct1_r2_events.txt"
rawEEG_2 = np.load(file)
eventos_2 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_2\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_2\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_3 = np.load(file)
eventos_3 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_2\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_2\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
rawEEG_4 = np.load(file)
eventos_4 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_2\eegdata\sesion2\sn2_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_2\eegdata\sesion2\sn2_ts0_ct0_r1_events.txt"
rawEEG_5 = np.load(file)
eventos_5 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_2\eegdata\sesion2\sn2_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_2\eegdata\sesion2\sn2_ts0_ct1_r1_events.txt"
rawEEG_6 = np.load(file)
eventos_6 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_3\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_3\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_7 = np.load(file)
eventos_7 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_3\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_3\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
rawEEG_8 = np.load(file)
eventos_8 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_4\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_4\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_9 = np.load(file)
eventos_9 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_4\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_4\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
rawEEG_10 = np.load(file)
eventos_10 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_5\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_5\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_11 = np.load(file)
eventos_11 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_5\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_5\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
rawEEG_12 = np.load(file)
eventos_12 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_6\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_6\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_13 = np.load(file)
eventos_13 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_6\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_6\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
rawEEG_14 = np.load(file)
eventos_14 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_7\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_7\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_15 = np.load(file)
eventos_15 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_7\eegdata\sesion1\sn1_ts0_ct1_r1.npy"
eventosFile = "data\sujeto_7\eegdata\sesion1\sn1_ts0_ct1_r1_events.txt"
rawEEG_16 = np.load(file)
eventos_16 = pd.read_csv(eventosFile, sep = ",")

#Creamos objetos para manejar los trials
th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0, tmax = 3, reject=None, sample_rate=250., trialsToRemove = [29,30])
th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0, tmax = 3, reject=None, sample_rate=250., trialsToRemove = [1])
th_3 = TrialsHandler(rawEEG_3, eventos_3, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_4 = TrialsHandler(rawEEG_4, eventos_4, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_5 = TrialsHandler(rawEEG_5, eventos_5, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_6 = TrialsHandler(rawEEG_6, eventos_6, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_7 = TrialsHandler(rawEEG_7, eventos_7, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_8 = TrialsHandler(rawEEG_8, eventos_8, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_9 = TrialsHandler(rawEEG_9, eventos_9, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_10 = TrialsHandler(rawEEG_10, eventos_10, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_11 = TrialsHandler(rawEEG_11, eventos_11, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_12 = TrialsHandler(rawEEG_12, eventos_12, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_13 = TrialsHandler(rawEEG_13, eventos_13, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_14 = TrialsHandler(rawEEG_14, eventos_14, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_15 = TrialsHandler(rawEEG_15, eventos_15, tinit = 0, tmax = 3, reject=None, sample_rate=250.)
th_16 = TrialsHandler(rawEEG_16, eventos_16, tinit = 0, tmax = 3, reject=None, sample_rate=250.)

dataConcatenada = Concatenate([th_1, th_2,th_3,th_4,th_5,th_6,th_7,th_8,th_9,th_10,th_11,th_12,th_13,th_14,th_15,th_16])#concatenamos datos

filtro = Filter(highcut = 16)
dataConcatenada.trials = filtro.transform(dataConcatenada.trials)

print(dataConcatenada.trials.shape)
print(dataConcatenada.labels.shape)

# Guardar la matriz en un archivo .npy
np.save('data.npy', dataConcatenada.trials)
np.save('labels.npy', dataConcatenada.labels)
