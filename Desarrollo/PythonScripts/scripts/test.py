import numpy as np
import scipy.io
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# from SignalProcessor import Filter, FeatureExtractor, Classifier
from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.Classifier import Classifier

"""
Descripción de los datos en: https://bbci.de/competition/iv/desc_1.html

Format of the Data
Given are continuous signals of 59 EEG channels and, for the calibration data, markers that indicate the time points of cue presentation
and the corresponding target classes.

Data are provided in Matlab format (*.mat) containing variables:

cnt: the continuous EEG signals, size [time x channels]. The array is stored in datatype INT16. To convert it to uV values,
use cnt= 0.1*double(cnt); in Matlab.

mrk: structure of target cue information with fields (the file of evaluation data does not contain this variable)
    pos: vector of positions of the cue in the EEG signals given in unit sample, length #cues
    y: vector of target classes (-1 for class one or 1 for class two), length #cues

nfo: structure providing additional information with fields
    fs: sampling rate,
    clab: cell array of channel labels,
    classes: cell array of the names of the motor imagery classes,
    xpos: x-position of electrodes in a 2d-projection,
    ypos: y-position of electrodes in a 2d-projection.


IMPORTANTE!! Leer el "Experimental Setup" en https://bbci.de/competition/iv/desc_1.html

"""

"""Acerca de las clases:

These data sets were recorded from healthy subjects. In the whole session motor imagery was
performed without feedback. For each subject two classes of motor imagery were selected from the three classes
left hand, right hand, and foot (side chosen by the subject; optionally also both feet)."""

def get_names(path):
    import os
    names = []
    for file in os.listdir(path):
        if file.endswith(".mat"):
            names.append(file)
    return names

def prepareData(rawData, path = "dataset/"):
    """El argumento datos contiene una lista de diccionarios. Cada diccionario contiene los EEG e información referente al registro.
    La función devuelve un numpy array con todos los EEGs y un diccionario que contiene diferente información que se utilizará para 
    análisis de los registros de EEG"""

    dataReady = dict()

    for i, data in enumerate(rawData):
        sample_rate = data['nfo']['fs'][0][0][0]
        EEG = data['cnt'].T
        nchannels, nsamples = EEG.shape

        channel_names = [s[0] for s in data['nfo']['clab'][0][0][0]]
        event_starting = data['mrk'][0][0][0]
        event_codes = data['mrk'][0][0][1]
        labels = np.zeros((1, nsamples), int)
        labels[0, event_starting] = event_codes

        cl_lab = [s[0] for s in data['nfo']['classes'][0][0][0]]
        cl1 = cl_lab[0]
        cl2 = cl_lab[1]
        nclasses = len(cl_lab)
        nevents = len(event_starting)

        xpos = data["nfo"]["xpos"][0,0]
        ypos = data["nfo"]["ypos"][0,0]

        dataReady[f"subject{i+1}"] = {
            "eeg": EEG,
            "sample_rate": sample_rate,
            "nchannels": nchannels,
            "nsamples": nsamples,
            "channelsNames": channel_names,
            "event_starting": event_starting,
            "event_codes": event_codes,
            "labels": labels,
            "class1": cl1,
            "class2": cl2,
            "nclasses": nclasses,
            "nevents": nevents,
            "xpos": xpos,
            "ypos": ypos,
        } 

    return dataReady


#La siguiente función hace exactamente lo mismo que la clase FeatureExtractor.py de nuestro sistema
def transform(signal):
    """Function to apply the filters to the signal.
    -signal: It is the signal in a numpy array of the form [channels, samples]."""

    #NOTA: Debemos evaluar si implementamos un CSP para seleccionar los mejores canales y luego aplicamos la transformada de Hilbert

    #Aplicamos la transformada de Hilbert
    transformedSignal = hilbert(signal, axis=1) #trnasformada de Hilbert
    power = np.abs(transformedSignal)**2 #Calculamos la potencia de la señal
    alphaPower = power[:, 8:13] #Potencia media en la banda alfa
    betaPower = power[:, 13:30] #Potencia media en la banda beta
    features = np.hstack((alphaPower, betaPower))#apilamos las características

    return features


def getTrials(EEG, cl_lab, event_codes, event_onsets, nchannels,
              w1 = 0.5, w2 = 2.5, sample_rate = 100.0):
    """Obtenemos los trials a partir del EEG
    
    - EEE: numpy array con la señal EEG en la forma [channels, samples]
    - cl_lab: lista con los nombres de las clases
    - event_codes: numpy array con los códigos de los eventos
    - event_onsets: numpy array con los comienzos de los eventos
    - nchannels: número de canales
    - w1: tiempo en segundos antes del evento
    - w2: tiempo en segundos después del evento
    - sample_rate: frecuencia de muestreo de la señal EEG"""

    # Diccionario con los datos de los registros EEG
    trials = {}

    #La ventana de tiempo se define en segundos. En este caso, 0.5 segundos antes del evento y 2.5 segundos después del evento
    win = np.arange(int(w1*sample_rate), int(w2*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extraemos los comienzos de los eventos de la clase cl
        cl_onsets = event_onsets[event_codes == code]
        
        # Guardamos memoria para los trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extraemos cada trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEG[:, win+onset]

    #los datos dentro de trials son de la forma [channels, samples, trials]
    return trials



"""INICIAMOS ESTUDIO DE LOS DATOS"""

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

channels = ["C3", "Cz", "C4"]
c3 = sujeto1["channelsNames"].index("C3")
cz = sujeto1["channelsNames"].index("Cz")
c4 = sujeto1["channelsNames"].index("C4")

eeg = eeg[[c3, cz, c4], :]

#Aplicamos el filtro
filtro = Filter()
eeg_filtered = filtro.fit_transform(eeg, lowcut = 6.0, highcut = 30.0, notch_freq = 50.0, notch_width = 2.0, sample_rate = 100)

#graficamos un canal de la señal original y la señal filtrada en un intervalo de tiempo
plt.figure(figsize=(10, 5))
plt.plot(eeg[0, 1000:2000], label = "Original")
plt.plot(eeg_filtered[0, 1000:2000], label = "Filtered")
plt.legend()
plt.show()

#Obtenemos los trials
trials = getTrials(eeg_filtered, [sujeto1["class1"], sujeto1["class2"]], sujeto1["event_codes"], sujeto1["event_starting"], len(channels))

clase1 = sujeto1["class1"]
clase2 = sujeto1["class2"]
print("La forma de los trials de la clase", clase1, "es:", trials[clase1].shape)
print("La forma de los trials de la clase", clase2, "es:", trials[clase2].shape)

#graficamos un trial de cada clase y un canal con leyenda
plt.figure(figsize=(10, 5))
plt.plot(trials[clase1][0, :, 0], label = clase1)
plt.plot(trials[clase2][0, :, 0], label = clase2)
plt.legend()
plt.title("Señales filtradas de la clase "+clase1+" y "+clase2+" en el canal C3")
plt.show()

#extracting features using FeatureExtractor.py
featureExtractor = FeatureExtractor(method = "hilbert")
featurescl1 = featureExtractor.fit_transform(trials[clase1])
featurescl2 = featureExtractor.fit_transform(trials[clase2])

trial = 40
#plotting the features for one trial and the two classes. Using title and legend
plt.figure(figsize=(10, 5))
plt.plot(featurescl1[0, :, trial-1], label = clase1)
plt.plot(featurescl2[0, :, trial-1], label = clase2)
plt.legend()
plt.title(f"Potencia en la banda alfa y beta para el trial {trial} de la clase {clase1} y {clase2}")
plt.show()

