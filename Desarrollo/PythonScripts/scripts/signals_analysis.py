from SignalProcessor.Filter import Filter
from TrialsHandler.TrialsHandler import TrialsHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filter = Filter(8, 12, 50, 2, 250., 2)

file = "data\pablo_testing\eegdata\sesion1\sn1_ts0_ct1_r2.npy"
rawEEG = np.load(file)

eventosFile = "data\pablo_testing\eegdata\sesion1\sn1_ts0_ct1_r2_events.txt"
eventos = pd.read_csv(eventosFile, sep = ",")

channelsName = ["C3", "CZ", "C4"]

tinit = 1 #el tiempo de inicio se considera ANTES del cue
tmax = 6 #el tiempo máximo debe considerarse entre el cue y el final del trial

trialhandler = TrialsHandler(rawEEG, eventos, tinit = tinit, tmax = tmax, reject=None, sample_rate=250.)
trialhandler.eventos.head()

classesName, classesLabel = trialhandler.classesName
labels = trialhandler.labels
raw_trials = filter.fit_transform(trialhandler.trials) #extraemos los trials y los filtramos

trials_x_clase = np.zeros((len(classesName),int(raw_trials.shape[0]/len(classesName)), raw_trials.shape[1], raw_trials.shape[2]))

#Por cada label dentro de classNames, filtramos los trials y lo guardamos en la posición correspondiente dentro de trials
for label in classesLabel:
    trials_x_clase[label-1,:,:,:] = raw_trials[labels == label]

### Gráfico de la señal sin filtrar.

trial_n = 1 #Selecciono el trial que quiero clasificar
label = 2 #Selecciono el label que quiero filtrar

eventos.index = eventos["trialNumber"]
min_tinit = eventos["startingTime"].min()
startingTime = eventos.loc[trial_n]["startingTime"]
cue_duration = eventos.loc[trial_n]["cueDuration"]
finish_time = eventos.loc[trial_n]["finishDuration"]

trozo_inicial = startingTime - tinit
trozo_final = tmax - cue_duration

#eje temporal. El mismo va desde -tinit hasta tmax
t = np.arange(-tinit, tmax, 1/250.)

plt.style.use('default')
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, trials_x_clase[label-1,trial_n-1, i, :])
    #linea vertical en el tiempo de inicio del cue. Linea punteada
    plt.axvline(x = 0, color="#656ccf", linestyle="--")
    #agrego un rectangulo de color verde con fondo transparente entre el tiempo de inicio del cue y el tiempo del cue
    plt.axvspan(0, cue_duration, color="#65cf70", alpha=0.2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
    plt.title("Canal {}".format(channelsName[i]))

plt.suptitle(f"Señal para el trial {trial_n} - Clase {classesName[classesLabel.index(label)]}")
#dismiuimos el espacio entre subplots
plt.tight_layout()
plt.show()

## promedio los trials

trials_promedio = trials_x_clase.mean(axis=1)

##Grafico tres canales en tres subplots
plt.style.use('default')
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, trials_promedio[label-1, i, :])
    #linea vertical en el tiempo de inicio del cue. Linea punteada
    plt.axvline(x = 0, color="#656ccf", linestyle="--")
    #agrego un rectangulo de color verde con fondo transparente entre el tiempo de inicio del cue y el tiempo del cue
    plt.axvspan(0, cue_duration, color="#65cf70", alpha=0.2)
    #cambiamos el color del fondo de toda la figura a gris
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
    plt.title("Canal {}".format(channelsName[i]))

plt.suptitle(f"Promedio sobre {trial_n} - Clase {classesName[classesLabel.index(label)]}")
#dismiuimos el espacio entre subplots
plt.tight_layout()
plt.show()
