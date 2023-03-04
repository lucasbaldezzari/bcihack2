import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# from SignalProcessor import Filter, FeatureExtractor, Classifier
from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.Classifier import Classifier

from tools import get_names, prepareData, getTrials, plot_signals

"""INICIAMOS ESTUDIO DE LOS DATOS"""

#Descripción de los datos en: https://bbci.de/competition/iv/desc_1.html
#IMPORTANTE!! Leer el "Experimental Setup" en https://bbci.de/competition/iv/desc_1.html

path = "__datasetsForTesting/"

files = get_names(path)

datos = []
for file in files: #nos quedamos solo con los primeros dos archivos
    datos.append(scipy.io.loadmat(path+file, struct_as_record = True))

datoslistos = prepareData(datos)

## A continuación haremos un análisis considerando sólo los datos del sujeto 1
sujeto1 = datoslistos["subject1"]

eeg = sujeto1["eeg"]

print("Cantidad de datos de EEG: ", eeg.shape)
print("Frecuencia de muestreo: ", sujeto1["sample_rate"])
print("Número de canales:", sujeto1["nchannels"])
print("Nombre de los canales:", sujeto1["channelsNames"])
print("Número de muestras:", sujeto1["nsamples"])
print("Códigos de los eventos:", sujeto1["event_codes"])
print("Etiquetas de las clases:", sujeto1["labels"])
print("Cantidad de clases:", sujeto1["nclasses"])
print("La clase 1 es:", sujeto1["class1"])
print("La clase 2 es:", sujeto1["class2"])

channels = ["C3", "Cz", "C4"]
c3 = sujeto1["channelsNames"].index("C3")
cz = sujeto1["channelsNames"].index("Cz")
c4 = sujeto1["channelsNames"].index("C4")

eeg = eeg[[c3, cz, c4], :] #nos quedamos solo con los canales C3, Cz y C4

#Aplicamos filtro a la señal
filtro = Filter()
eeg_filtered = filtro.fit_transform(eeg, lowcut = 6.0, highcut = 30.0, notch_freq = 50.0, notch_width = 2.0, sample_rate = 100)

#Gráficamos un trozo de las señales filtradas y no filtradas
plot_signals([eeg[:, 0:1000][0], eeg_filtered[:, 0:1000][0]], title="Señales filtradas y no filtradas",
             legend=["Señal no filtrada", "Señal filtrada"])

#Dividimos la señal en trials considerando los event_starting
trials = getTrials(eeg_filtered, [sujeto1["class1"], sujeto1["class2"]], sujeto1["event_codes"], sujeto1["event_starting"], len(channels),
                   w1=0.2, w2=2.5, sample_rate=100)

clase1 = sujeto1["class1"]
clase2 = sujeto1["class2"]
print("La forma de los trials de la clase", clase1, "es:", trials[clase1].shape)
print("La forma de los trials de la clase", clase2, "es:", trials[clase2].shape)

#Graficamos un trial de cada clase para un canal
plot_signals([trials[clase1][0, :, 0], trials[clase2][0, :, 0]], title="Señales de la clase 1 y 2",
                legend=["Señal de la clase 1", "Señal de la clase 2"])


#Extraemos las características de los trials usando FeatureExtractor.py
featureExtractor = FeatureExtractor(method = "hilbert")
featurescl1 = featureExtractor.fit_transform(trials[clase1])
featurescl2 = featureExtractor.fit_transform(trials[clase2])

#Plotting the features using plot_signals for one trial and one channel
plot_signals([featurescl1[0, :, 0], featurescl2[0, :, 0]], title="Características de la clase 1 y 2 - Hilbert",
                legend=["Features Cl1", "Features Cl1"])

#Extraemos características usando psd
featureExtractor = FeatureExtractor(method = "psd")
featurescl1_psd = featureExtractor.fit_transform(trials[clase1])
featurescl2_psd = featureExtractor.fit_transform(trials[clase2])

#Plotting the features using plot_signals for one trial and one channel
plot_signals([featurescl1_psd[0, :, 0], featurescl2_psd[0, :, 0]], title="Características de la clase 1 y 2 - PSD",
                legend=["Features Cl1", "Features Cl1"])