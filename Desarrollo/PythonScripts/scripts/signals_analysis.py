from SignalProcessor.Filter import Filter
from TrialsHandler.TrialsHandler import TrialsHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filter = Filter(5, 48, 50, 2, 250., 2)

file = "data\mordiendo\eegdata\sesion1\sn1_ts0_ct1_r2.npy"
rawEEG = np.load(file)

eventosFile = "data\mordiendo\eegdata\sesion1\sn1_ts0_ct1_r2_events.txt"
eventos = pd.read_csv(eventosFile, sep = ",")

tinit = 1#el tiempo de inicio se considera ANTES del cue
tmax = 6
trialhandler = TrialsHandler(rawEEG, eventos, tinit = tinit, tmax = tmax, reject=None, sample_rate=250.)

trials = trialhandler.getTrials()
labels = trialhandler.getLabels()
trials.shape
labels

#Clase 1 = Morder, Clase 2 = Rest
trials_c1 = trials[labels == 1]
trials_c2 = trials[labels == 2]

#filtramos
trials_c1 = filter.fit_transform(trials_c1)
trials_c2 = filter.fit_transform(trials_c2)

## Grafico de la señal en los tres canales para el trial 1 de trials_c1
## Coloco una linea vertical en el tiempo tinicio
## Cada canal en un suplot
## Uso un for para crear los subplots

#eje temporal. El mismo va desde -tinit hasta tmax
t = np.arange(-tinit, tmax, 1/250.)

trial_n = 1 #trial 1

##seteamos index de eventos a los trialNumber
##ggplot style

eventos.index = eventos["trialNumber"]
min_tinit = eventos["startingTime"].min()
startingTime = eventos.loc[trial_n]["startingTime"]
cue_duration = eventos.loc[trial_n]["cueDuration"]
finish_time = eventos.loc[trial_n]["finishDuration"]

trozo_inicial = startingTime - min_tinit
trozo_final = tmax - cue_duration

plt.style.use('seaborn')
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, trials_c1[trial_n-1, i, :])
    #linea vertical en el tiempo de inicio del cue. Linea punteada
    plt.axvline(x=trozo_inicial, color="#656ccf", linestyle="--")
    #agrego un rectangulo de color verde con fondo transparente entre el tiempo de inicio del cue y el tiempo del cue
    plt.axvspan(trozo_inicial, trozo_inicial + cue_duration, color="#65cf70", alpha=0.2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
    plt.title("Canal {}".format(i+1))

plt.suptitle("Señal de los tres canales para el trial 1 de trials_c1")
#dismiuimos el espacio entre subplots
plt.tight_layout()
plt.show()

## repito las gráficas pero para la clase 2

plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t, trials_c2[trial_n-1, i, :])
    #linea vertical en el tiempo de inicio del cue. Linea punteada
    plt.axvline(x=trozo_inicial, color="#656ccf", linestyle="--")
    #agrego un rectangulo de color verde con fondo transparente entre el tiempo de inicio del cue y el tiempo del cue
    plt.axvspan(trozo_inicial, trozo_inicial + cue_duration, color="#65cf70", alpha=0.2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (uV)")
    plt.title("Canal {}".format(i+1))

plt.suptitle("Señal de los tres canales para el trial 1 de trials_c2")
#dismiuimos el espacio entre subplots
plt.tight_layout()
plt.show()
