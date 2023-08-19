import numpy as np
import pandas as pd

from TrialsHandler.TrialsHandler import TrialsHandler
from TrialsHandler.Concatenate import Concatenate
# from TrialsHandler.TrialsHandlerv2 import ModifiedTrialsHandler

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

import os

def listar_archivos(directorio_raiz, filtro="", subcarpeta="eegdata", carpeta_excluida="sesion0"):
    rawEEGs = []
    events = []

    # Recorrer el directorio raíz y sus subdirectorios
    for dirpath, dirnames, filenames in os.walk(directorio_raiz):
        # Verificar si el nombre del directorio actual contiene la subcadena de filtro
        if filtro in os.path.basename(dirpath):
            # Construir la ruta hacia la subcarpeta
            ruta_subcarpeta = os.path.join(dirpath, subcarpeta)
            
            # Verificar si la subcarpeta existe
            if os.path.exists(ruta_subcarpeta):
                # Recorrer la subcarpeta y sus subdirectorios
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(ruta_subcarpeta):
                    # Evitar entrar en la carpeta excluida
                    if carpeta_excluida in sub_dirnames:
                        sub_dirnames.remove(carpeta_excluida)
                    
                    for archivo in sub_filenames:
                        ruta_completa = os.path.join(sub_dirpath, archivo)
                        if archivo.endswith('.npy'):
                            rawEEGs.append([[dirpath[-1]],[np.load(ruta_completa)]])
                            
                        elif archivo.endswith('.txt'):
                            events.append(pd.read_csv(ruta_completa, sep = ","))
                        
    return rawEEGs, events

EEGs, eventos = listar_archivos('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data', "suj")

print(EEGs.shape)

# # EEGs = np.load('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data/sujeto_1/eegdata/sesion1/sn1_ts0_ct0_r1.npy')
# # eventos = pd.read_csv('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data/sujeto_1/eegdata/sesion1/sn1_ts0_ct0_r1_events.txt', sep = ",")

# trialsHand = []

# for eeg, evento in zip(EEGs, eventos):
#     trialsHand.append(TrialsHandler(eeg, evento, tinit=0, tmax=2, reject=None, sample_rate=250.))

# # trialsHand.append(ModifiedTrialsHandler(EEGs, eventos, tinit=0, tmax=10, reject=None, sample_rate=250.))

# dataConcatenada = Concatenate(trialsHand)#concatenamos datos

# trials = dataConcatenada.trials
# #me quedo con channelsSelected
# labels = dataConcatenada.labels

# # eeg_train, eeg_test, labels_train, labels_test = train_test_split(trials, labels, test_size=0.2, stratify=labels)
# # eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_train, labels_train, test_size=0.2, stratify=labels_train)

# fm = 250. #frecuencia de muestreo
# filter = Filter(lowcut=5, highcut=18, notch_freq=50.0, notch_width=2, sample_rate=fm, axisToCompute=2, padlen=None, order=4)
# #Creamos un CSPMulticlass - Método ovo (one vs one)
# cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = 5, reg = 0.01)
# featureExtractor = FeatureExtractor(method = "welch", sample_rate = fm, axisToCompute=2, band_values=[8,12])
# # ravelTransformer = RavelTransformer()

# #Instanciamos un LDA
# # lda = LDA() #instanciamos el clasificador LDA

# ### ********** Creamos el pipeline para LDA **********

# pipeline_lda = Pipeline([
#     ('pasabanda', filter),
#     ('cspmulticlase', cspmulticlass)
# ])

# # filtro = Filter(highcut = 16)
# pipeline_lda.fit(trials, labels)
# dataConcatenada.trials = pipeline_lda.transform(trials)

# print(dataConcatenada.trials.shape)
# print(dataConcatenada.labels.shape)

# # Guardar la matriz en un archivo .npy
# np.save('data.npy', dataConcatenada.trials)
# np.save('labels.npy', dataConcatenada.labels)