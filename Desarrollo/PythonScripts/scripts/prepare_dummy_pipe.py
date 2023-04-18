import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import json

from TrialsHandler.TrialsHandler import TrialsHandler
from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.CSPMulticlass import CSPMulticlass
from SignalProcessor.RavelTransformer import RavelTransformer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import pickle




"""Usaremos el registro de la Syntehtic board para entrenar y usar el pipeline"""

file = file = "data/dummyTest/eegdata/sesion1/sn1_ts0_ct0_r1.npy"
eventosFile = "data/dummyTest/eegdata/sesion1/sn1_ts0_ct0_r1_events.txt"
configFile = "data/dummyTest/eegdata/sesion1/sn1_ts0_ct0_r1_config.json"
cspsFolder = "data/dummyTest/csps/"
classifiersFolder = "data/dummyTest/classifiers/"
pipelinesFolder = "data/dummyTest/pipelines/"


#cargamos archivo txt de eventos y lo pasamos a un dataframe
eventos = pd.read_csv(eventosFile, sep=",")

#Cargamos archivo de eeg
raw_eeg = np.load(file)
channels = [3,4,5] #canales de interés
raw_eeg = raw_eeg[channels]
print("raw_eeg shape:", raw_eeg.shape)

#cargamos parametros para extraer los nombres de las clases
config = json.load(open(configFile))
clasesNames = config["clasesNames"]


#debemos dividir la señal de EEG en trials. Cada trial es la suma del startingTime y el cueDuration
#Nos interesa quedarnos con el cueDuration.
#Utilizamos la frecuencia de muestreo para calcular la cantidad de muestras que representa el cueDuration

sample_rate = 250.
trialshandler = TrialsHandler(raw_eeg, eventos, tmin = 0.0, tmax = 1, reject=None, sample_rate = sample_rate)
trials = trialshandler.trials
labels = trialshandler.labels
print(trials.shape)
print(labels.shape)
print(labels[:10])

#Separamos los datos en train, test y validación
eeg_train, eeg_test, labels_train, labels_test = train_test_split(trials, labels, test_size=0.2, random_state=1)
eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_train, labels_train, test_size=0.2, random_state=1)


## ******************************
## Creamos el pipeline
## ******************************

#Creamos un filtro pasabanda
pasabanda = Filter(lowcut=8.0, highcut=28.0, notch_freq=50.0, notch_width=2.0, sample_rate=100.0)

#Creamos un CSPMulticlass - Método ovo (one vs one)
cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = len(np.unique(labels)), reg=None, log=None, norm_trace=False)
print(f"Cantidad de filtros CSP a entrenar: {len(cspmulticlass.csplist)}")

featureExtractor = FeatureExtractor(method = "welch", sample_rate=100., axisToCompute=2)

ravelTransformer = RavelTransformer()
scaler = StandardScaler()

#Vamos a probar un LDA
pipeline_lda = Pipeline([
    ('pasabanda', pasabanda),
    ('cspmulticlase', cspmulticlass),
    ('featureExtractor', featureExtractor),
    ('ravelTransformer', ravelTransformer),
    # ('scaler', scaler),
    ('lda', LinearDiscriminantAnalysis())
])


"""Análisis rápido con el pipeline"""
pipeline_lda.fit(eeg_train, labels_train)
print(pipeline_lda.score(eeg_test, labels_test)) #Score de 20%!!!! Recordemos que son datos dummy

# """Búsqueda de hiperparámetros con GridSearchCV"""
# #Definimos una grilla de parámetros para el GridSearch
# param_grid_lda = {
#     'pasabanda__lowcut': [8.0],
#     'pasabanda__highcut': [15.0],
#     'cspmulticlase__n_components': [2],
#     'featureExtractor__method': ['hilbert'],
#     'lda__solver': ['eigen'],   # LDA solver
#     'lda__shrinkage': [None, 'auto'],  # LDA shrinkage
#     }

# #Creamos el GridSearch para el LDA
# grid_lda = GridSearchCV(pipeline_lda, param_grid_lda, cv=5, n_jobs=-1)
# grid_lda.fit(eeg_train, labels_train)


# #Nos quedamos con el mejor estimador
# best_stimator_lda = grid_lda.best_estimator_

# #Usamos el mejor estimador para predecir sobre los datos de test
# best_stimator_lda.fit(eeg_test, labels_test)

# #Usamos el mejor estimador para predecir sobre los datos de validación
# y_pred = best_stimator_lda.predict(eeg_val)

# #Calculamos la matriz de confusión
# confusion_matrix(labels_val, y_pred)

# #Calculamos el accuracy
# print(accuracy_score(labels_val, y_pred))

# #usamos el modelo para predecir sobre un nuevo trial (usando un trial de la clase 1 de los datos de validación)
# trial = eeg_val[9].reshape(1, eeg_val.shape[1], eeg_val.shape[2])
# print(trial.shape)

# #predecimos
# y_pred = best_stimator_lda.predict(trial[:,:,:750])
# print(y_pred)

# #Guardamos el best_stimator_lda en la carpeta pipelines
# filename = pipelinesFolder + "pipeline.pkl"
# pickle.dump(best_stimator_lda, open(filename, 'wb'))

#### SVM ###########
#Ahora probamos un SVM
pipeline_svm = Pipeline([
    ('pasabanda', pasabanda),
    ('cspmulticlase', cspmulticlass),
    ('featureExtractor', featureExtractor),
    ('ravelTransformer', ravelTransformer),
    ('scaler', scaler),
    ('svm', SVC())
])

"""Análisis rápido con el pipeline"""
pipeline_svm.fit(eeg_train, labels_train)
print(pipeline_svm.score(eeg_test, labels_test)) #Score de 20%!!!! Recordemos que son datos dummy

# Parámetros del grid para SVM
param_grid_svm = {
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto']
}

#Creamos el GridSearch para el LDA
grid_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, n_jobs=-1)
grid_svm.fit(eeg_train, labels_train)

#Nos quedamos con el mejor estimador
best_estimator_svm = grid_svm.best_estimator_

#Usamos el mejor estimador para predecir sobre los datos de test
best_estimator_svm.fit(eeg_test, labels_test)

#Usamos el mejor estimador para predecir sobre los datos de validación
y_pred = best_estimator_svm.predict(eeg_val)

#Calculamos la matriz de confusión

confusion_matrix(labels_val, y_pred)

#Calculamos el accuracy
print(accuracy_score(labels_val, y_pred))

#usamos el modelo para predecir sobre un nuevo trial (usando un trial de la clase 1 de los datos de validación)
trial = eeg_val[1].reshape(1, eeg_val.shape[1], eeg_val.shape[2])
print(trial.shape)
#predecimos
y_pred = best_estimator_svm.predict(trial)
print(y_pred)

#Guardamos el best_stimator_lda en la carpeta pipelines
filename = pipelinesFolder + "best_estimator_svm.pkl"
pickle.dump(best_estimator_svm, open(filename, 'wb'))