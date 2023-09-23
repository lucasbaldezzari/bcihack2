"""Script para generar un pipeline para evaluar diferentes combinaciones de hiperparámetros
y obtener el mejor modelo para clasificar las señales de EGG.
"""

import numpy as np
import pandas as pd

from TrialsHandler.TrialsHandler import TrialsHandler
from TrialsHandler.Concatenate import Concatenate

from SignalProcessor.Filter import Filter
from SignalProcessor.CSPMulticlass import CSPMulticlass

### ********** Cargamos los datos **********
file = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
rawEEG_1 = np.load(file)
eventos_1 = pd.read_csv(eventosFile, sep = ",")

file = "data\sujeto_1\eegdata\sesion2\sn2_ts0_ct0_r1.npy"
eventosFile = "data\sujeto_1\eegdata\sesion2\sn2_ts0_ct0_r1_events.txt"
rawEEG_2 = np.load(file)
eventos_2 = pd.read_csv(eventosFile, sep = ",")

#Creamos objetos para manejar los trials
th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0, tmax = 3, reject=None, sample_rate=250., trialsToRemove = [29,30])
th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0, tmax = 3, reject=None, sample_rate=250.)

dataConcatenada = Concatenate([th_1, th_2])#concatenamos datos

filtro = Filter(highcut = 16)
dataConcatenada.trials = filtro.transform(dataConcatenada.trials)

csp = CSPMulticlass(n_components=2, method = "ovo", n_classes = 5, reg = 0.01)
csp.fit(dataConcatenada.trials, dataConcatenada.labels)
dataConcatenada.trials = csp.transform(dataConcatenada.trials)

print(dataConcatenada.trials.shape)
print(dataConcatenada.labels.shape)

# Guardar la matriz en un archivo .npy
np.save('data.npy', dataConcatenada.trials)
np.save('labels.npy', dataConcatenada.labels)
