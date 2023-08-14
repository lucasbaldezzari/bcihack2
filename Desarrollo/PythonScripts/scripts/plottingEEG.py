import numpy as np
import pandas as pd
from SignalProcessor.Filter import Filter
from EEGPlotter.EEGPlotter import EEGPlotter
from TrialsHandler.TrialsHandler import TrialsHandler

fm = 250.
filtro = Filter(8, 18, 50, 2, fm, 1)

## cargamos archivos
file = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct1_r2.npy"
rawEEG = np.load(file)

eeg = filtro.fit_transform(rawEEG)

eventosFile = "data\sujeto_1\eegdata\sesion1\sn1_ts0_ct1_r2_events.txt"
eventos = pd.read_csv(eventosFile, sep = ",")

##convertimos la coluna eventos["trialTime(legible)"][0] a datetime
tiempos = pd.to_datetime(eventos["trialTime(legible)"])

#nos quedamos con los minutos y segundos y se lo vamos sumando a medida que recorremos los trials
tiempos = tiempos.dt.minute*60 + tiempos.dt.second
tiempos = tiempos - tiempos[0]

tinit = 0.
tmax = 4.

## Instanciamos la clase TrialsHandler para extraer los trials, labels, nombre de clases, etc.
trialhandler = TrialsHandler(rawEEG, eventos,
                             tinit = tinit, tmax = tmax,
                             reject=None, sample_rate=fm,
                             trialsToRemove=None)

labels = trialhandler.labels
#agregamos un número creciente de 1 al largo de lables delante de cada label
labels = [str(i)+"-"+"C"+str(labels[i-1]) for i in range(1,len(labels)+1)]
labels

ini_trial = 55
final_trial = 75
trial_duration = 10

ti = int(ini_trial*trial_duration*fm)
tf = int(final_trial*trial_duration*fm)

trials_times = tiempos.values[ini_trial:final_trial] - tiempos.values[ini_trial]

paso = 2 #segundos
window_size = 10 #segundos
eeg_plotter = EEGPlotter(eeg[:,ti:tf], fm, paso, window_size,task_window = (3,4),
                         labels = labels[ini_trial:final_trial], trials_start = trials_times)

eeg_plotter.plot()