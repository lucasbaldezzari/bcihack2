import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# from SignalProcessor import Filter, FeatureExtractor, Classifier
from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.CSPMulticlass import CSPMulticlass
from SignalProcessor.RavelTransformer import RavelTransformer

from tools import get_names, prepareData, getTrials, plot_signals

"""INICIAMOS ESTUDIO DE LOS DATOS"""

#Descripción de los datos en: https://bbci.de/competition/iv/desc_1.html
#IMPORTANTE!! Leer el "Experimental Setup" en https://bbci.de/competition/iv/desc_1.html

path = "SignalProcessor/testData/"

files = get_names(path)

datos = []
for file in files: #nos quedamos solo con los primeros dos archivos
    datos.append(scipy.io.loadmat(path+file, struct_as_record = True))

datoslistos = prepareData(datos)

## A continuación haremos un análisis considerando sólo los datos del sujeto 1
sujeto1 = datoslistos["subject1"]

#extraemos el eeg
eeg = sujeto1["eeg"] #eeg sin filtrar

#imprimios algunos datos
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

#algunos canales de interés
channels = ["C3", "Cz", "C4"]
c3 = sujeto1["channelsNames"].index("C3")
cz = sujeto1["channelsNames"].index("Cz")
c4 = sujeto1["channelsNames"].index("C4")

#Dividimos la señal en trials considerando los event_starting
trials = getTrials(eeg, [sujeto1["class1"], sujeto1["class2"]], sujeto1["event_codes"], sujeto1["event_starting"], 59,
                   w1=-0.5, w2=2.5, sample_rate=100)

clase1 = sujeto1["class1"]
clase2 = sujeto1["class2"]
print("La forma de los trials de la clase", clase1, "es:", trials[clase1].shape)
print("La forma de los trials de la clase", clase2, "es:", trials[clase2].shape)

# #saving the trials in npy files for each class
# np.save("SignalProcessor/testData/noisy_eeg_classLeft.npy", trials[clase1])
# np.save("SignalProcessor/testData/noisy_eeg_classRight.npy", trials[clase2])

#Graficamos el trial 50 de la clase 1 y la clase 2 para el canal c3
t1 = -0.5
t2 = 2.5
sample_rate = 100
tline = np.arange(t1, t2, 1/sample_rate)

## Concatemaos los trials de las dos clases
#Contactemos los trials de cada clase en un sólo array
eegmatrix = np.concatenate((trials[clase1],trials[clase2]), axis=0) #importante el orden con el que concatenamos
print(eegmatrix.shape) #[ n_trials (o n_epochs), n_channels, n_samples]

## generamos las labels para cada clase
class_info = {1: "left", 2: "right"} #diccionario para identificar clases. El orden se corresponde con lo que hay eneegmatrix
n_clases = len(list(class_info.keys()))

#genero las labels
n_trials = trials[clase1].shape[0]
totalTrials = eegmatrix.shape[0]
labels = np.array([np.full(n_trials, label) for label in class_info.keys()]).reshape(totalTrials)
print(labels.shape)
# print(labels) #las labels se DEBEN corresponder con el orden de los trials en eegmatrix


"""
*************** PIPELINE Y CLASIFICACIÓN ***************
Vamos a generar un pipeline para filtrar la señal, extraer características y clasificar.
"""

#Separamos los datos en train, test y validación
from sklearn.model_selection import train_test_split, GridSearchCV

eeg_train, eeg_test, labels_train, labels_test = train_test_split(eegmatrix, labels, test_size=0.2, random_state=1)
eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_train, labels_train, test_size=0.2, random_state=1)

## ******************************
## Creamos el pipeline
## ******************************

#Creamos un filtro pasabanda
pasabanda = Filter(lowcut=8.0, highcut=28.0, notch_freq=50.0, notch_width=2.0, sample_rate=100.0)

## Probamos el pasabanda
# eeg_train_filtered = pasabanda.fit_transform(eeg_train)

# ## graficamos el trial 1 de la clase 1 y la clase 2 para el canal c3 antes y después de filtrar
# plt.plot(tline,eeg_train[1, c3, :], label="Sin filtrar")
# plt.plot(tline,eeg_train_filtered[1, c3, :], label="Filtrado")
# plt.legend()
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Amplitud (uV)")
# plt.title("Señales de EEG para el canal C3 (filtrado)")
# plt.show()

#Creamos un CSPMulticlass - Método ovo (one vs one)
cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = len(np.unique(labels)), reg=None, log=None, norm_trace=False)
print(f"Cantidad de filtros CSP a entrenar: {len(cspmulticlass.csplist)}")

#Creamos un FeatureExtractor
featureExtractor = FeatureExtractor(method = "welch", sample_rate=100., axisToCompute=2)

#Transoformer para acomodar los datos de [n_trials, n_channels, n_samples] a [n_trials, n_componentes x n_samples] = [n_trials, n_features]
ravelTransformer = RavelTransformer()

#creamos el pipeline con un pasabanda, un cspmulticlase, un featureExtractor y un LDA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pipeline = Pipeline([
    ('pasabanda', pasabanda),
    ('cspmulticlase', cspmulticlass),
    ('featureExtractor', featureExtractor),
    ('ravelTransformer', ravelTransformer),
    ('lda', LinearDiscriminantAnalysis())
])

"""Análisis rápido con el pipeline"""
pipeline.fit(eeg_train, labels_train)
print(pipeline.score(eeg_test, labels_test))

"""Búsqueda de hiperparámetros con GridSearchCV"""
#Definimos una grilla de parámetros para el GridSearch
param_grid = {
    'pasabanda__lowcut': [8.0, 9.0],
    'pasabanda__highcut': [15.0, 28.0],
    'cspmulticlase__n_components': [2],
    'featureExtractor__method': ['hilbert','welch'],
    'lda__solver': ['svd', 'lsqr', 'eigen'],
    'lda__shrinkage': ['auto', None]
    }

#Creamos el GridSearch
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(eeg_train, labels_train)

#Nos quedamos con el mejor estimador
best_estimator = grid.best_estimator_

#Usamos el mejor estimador para predecir sobre los datos de test
best_estimator.fit(eeg_test, labels_test)

#Usamos el mejor estimador para predecir sobre los datos de validación
y_pred = best_estimator.predict(eeg_val)

#Calculamos la matriz de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix(labels_val, y_pred)

#Calculamos el accuracy
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(labels_val, y_pred))

#guardamos el mejor modelo con los hiparámetros encontrados en la carpeta models. Si no existe la carpeta, la creamos
import os
if not os.path.exists('models'):
    os.makedirs('models')

import pickle
filename = 'models/best_model.sav'
pickle.dump(best_estimator, open(filename, 'wb'))

#usamos el modelo para predecir sobre un nuevo trial (usando un trial de la clase 1 de los datos de validación)
trial = eeg_val[9].reshape(1, eeg_val.shape[1], eeg_val.shape[2])
print(trial.shape)

#predecimos
y_pred = best_estimator.predict(trial)
print(y_pred)



