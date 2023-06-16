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

tinit = -0.1 #el tiempo de inicio se considera ANTES del cue
tmax = 4.1 #el tiempo máximo debe considerarse entre el cue y el final del trial

trialhandler = TrialsHandler(rawEEG, eventos, tinit = tinit, tmax = tmax, reject=None, sample_rate=250.)
trialhandler.eventos.head()

classesName, classesLabel = trialhandler.classesName
labels = trialhandler.labels
raw_trials = filter.fit_transform(trialhandler.trials) #extraemos los trials y los filtramos

trials_x_clase = np.zeros((len(classesName),int(raw_trials.shape[0]/len(classesName)), raw_trials.shape[1], raw_trials.shape[2]))

#Por cada label dentro de classNames, filtramos los trials y lo guardamos en la posición correspondiente dentro de trials
for label in classesLabel:
    trials_x_clase[label-1,:,:,:] = raw_trials[labels == label]

trials_x_clase.shape

### Gráfico de la señal sin filtrar.

trial_n = 1 #Selecciono el trial que quiero clasificar
label = labels[trial_n-1] #Obtengo la etiqueta del trial

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

## Separo por clase
izq = trials_x_clase[0,:,:,:] #clase 1
der = trials_x_clase[1,:,:,:] #clase 2
ambasManos = trials_x_clase[2,:,:,:] #clase 3
ambosPies = trials_x_clase[3,:,:,:] #clase 4
rest = trials_x_clase[4,:,:,:] #clase 5

## Extraemos características

from SignalProcessor.FeatureExtractor import FeatureExtractor
fe_welch = FeatureExtractor(method="welch", sample_rate=250., axisToCompute=3) #instanciamos el extractor de características

trials_x_clase_welch = fe_welch.fit_transform(trials_x_clase)

trial = 3
indices = np.where((fe_welch.freqs >= 5) & (fe_welch.freqs <= 30))[0]

colors = ["#1e81b0", "#c13f48", "#65cf70", "#f5b041", "#5d6d7e"]

fig, axs = plt.subplots(len(channelsName), 1, figsize=(10, 8))
for i, channel in enumerate(channelsName):
    for j, clase in enumerate(classesName):
        axs[i].plot(fe_welch.freqs[indices], trials_x_clase_welch[j,trial - 1, i, :][indices], label=clase, linewidth=2, color = colors[j])
        #creamos un rectángulo para resaltar la banda mu
    axs[i].axvspan(8, 12, alpha=0.1, color='grey')
    axs[i].set_title(channel)
    axs[i].set_ylabel('Potencia (dB)')
    axs[i].set_xlabel('Frecuencia (Hz)')
    axs[i].legend()

#reduzco el padding entre subplots
plt.suptitle(f'EEG en la banda $mu$ para trial {trial} - Todas las clases')
plt.tight_layout()
plt.show()


#Extractor de características para datos de la forma [n_trials, n_channels, n_samples]
fe_welch = FeatureExtractor(method="welch", sample_rate=250., axisToCompute=2)

#extraemos las características para cada trial
izq_features = fe_welch.fit_transform(izq)
der_features = fe_welch.fit_transform(der)
ambasManos_features = fe_welch.fit_transform(ambasManos)
ambosPies_features = fe_welch.fit_transform(ambosPies)
rest_features = fe_welch.fit_transform(rest)

## Graficamos las características para dos clases
## Obtenemos las frecuencias entre 5 y 30 Hz a partir de fe.self.freqs
indices = np.where((fe_welch.freqs >= 5) & (fe_welch.freqs <= 30))[0]
#-1 left, 1 right
trial = 1
clase1 = 1
clase2 = 2

fig, axs = plt.subplots(len(channelsName), 1, figsize=(10, 8))
for i, channel in enumerate(channelsName):
    axs[i].plot(fe_welch.freqs[indices], trials_x_clase_welch[clase1-1, trial - 1, i, :][indices], linewidth=2, color = "#1e81b0")
    axs[i].plot(fe_welch.freqs[indices], trials_x_clase_welch[clase2-1, trial - 1, i, :][indices], linewidth=2, color = "#c13f48")
    #creamos un rectángulo para resaltar la banda mu
    axs[i].axvspan(8, 12, alpha=0.1, color='grey')
    axs[i].set_title(channel)
    axs[i].set_ylabel('Potencia (dB)')
    axs[i].set_xlabel('Frecuencia (Hz)')
    axs[i].legend([classesName[clase1-1], classesName[clase2-1]])

#reduzco el padding entre subplots
plt.suptitle(f'EEG en la banda $mu$ para trial {trial} - Clases {classesName[clase1-1]} y {classesName[clase1-2]}')
plt.tight_layout()
plt.show()


## Aplicamos CSP
##importamos la clase CSP
from SignalProcessor.CSPMulticlass import CSPMulticlass

cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = len(np.unique(labels)), reg = 0.01)

## Separamos en train y test
from sklearn.model_selection import train_test_split
#separamos en train y test. Balanceamos las clases
X_train, X_test, y_train, y_test = train_test_split(raw_trials, labels, test_size=0.2, random_state=42, stratify=labels)

#entrenamos el CSP
cspmulticlass.fit(X_train, y_train)

##Aplicamos el CSP a los datos de testeo
X_test_csp = cspmulticlass.transform(X_test)
X_test_csp.shape

## Extraemos las características de los datos de testeo con CSP
X_test_csp_wech = fe_welch.fit_transform(X_test_csp)

from SignalProcessor.RavelTransformer import RavelTransformer
raveltransformer = RavelTransformer() #instanciamos el raveltransformer

X_test_csp_wech_ravel = raveltransformer.fit_transform(X_test_csp_wech)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA() #instanciamos el clasificador LDA

X_train_csp = cspmulticlass.transform(X_train) #aplico el filtro CSP a los datos de entrenamiento
X_train_welch = fe_welch.fit_transform(X_train_csp) #aplico el extractor de características welch a los datos de entrenamiento
trials, componentes, frecuencias = X_train_welch.shape

X_train_welch = X_train_welch[:,:,[indices]].reshape(trials, componentes, len(indices))
X_train_welch_ravel = raveltransformer.fit_transform(X_train_welch)

lda.fit(X_train_welch_ravel, y_train) #entrenamos el clasificador

X_test_csp = cspmulticlass.transform(X_test) #aplico el filtro CSP a los datos de entrenamiento
X_test_welch = fe_welch.fit_transform(X_test_csp) #aplico el extractor de características welch a los datos de entrenamiento

trials, componentes, frecuencias = X_test_welch.shape
X_test_welch = X_test_welch[:,:,[indices]].reshape(trials, componentes, len(indices))
X_test_welch_ravel = raveltransformer.fit_transform(X_test_welch)

# plt.plot(X_test_welch_ravel[0,:])
# plt.plot(X_test_welch_ravel[1,:])
# plt.show()

#Evaluamos rápidamente el clasificador sobre los datos de test
print(f"El accuracy es {lda.score(X_test_welch_ravel, y_test)*100}%")

# #### REORDENANDO ****************
# indexes = np.array([np.arange(i, X_test_csp.shape[1], 2) for i in range(2)]).ravel()

# X_test_csp = X_test_csp[:,indexes,:] #reoordenamos los componentes

# X_test_welch = fe_welch.fit_transform(X_test_csp) #aplico el extractor de características welch a los datos de entrenamiento

# trials, componentes, frecuencias = X_test_welch.shape
# X_test_welch = X_test_welch[:,:,[indices]].reshape(trials, componentes, len(indices))
# X_test_welch_ravel = raveltransformer.fit_transform(X_test_welch)

# plt.plot(X_test_welch_ravel[1,:])
# plt.plot(X_test_welch_ravel[2,:])
# plt.plot(X_test_welch_ravel[3,:])
# plt.show()

# #Evaluamos rápidamente el clasificador sobre los datos de test
# print(f"El accuracy es {lda.score(X_test_welch_ravel, y_test)*100}%")
