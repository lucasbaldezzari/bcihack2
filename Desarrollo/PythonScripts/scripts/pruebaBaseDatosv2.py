import numpy as np
import pandas as pd

from TrialsHandler.TrialsHandler import TrialsHandler
from TrialsHandler.Concatenate import Concatenate

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
                            rawEEGs.append(np.load(ruta_completa))
                            
                        elif archivo.endswith('.txt'):
                            events.append(pd.read_csv(ruta_completa, sep = ","))
                        
    return rawEEGs, events

# directorio = input("Introduce el directorio raíz: ")
# filtro_nombre = input("Introduce la parte del nombre de la carpeta que quieres filtrar (deja en blanco para no filtrar): ")
EEGs, eventos = listar_archivos('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data', "sujeto")

trialsHand = []

for eeg, evento in zip(EEGs, eventos):
    trialsHand.append(TrialsHandler(eeg, evento, tinit = 0, tmax = 3, reject=None, sample_rate=250.))

dataConcatenada = Concatenate(trialsHand)#concatenamos datos

filtro = Filter(highcut = 16)
dataConcatenada.trials = filtro.transform(dataConcatenada.trials)

print(dataConcatenada.trials.shape)
print(dataConcatenada.labels.shape)

# Guardar la matriz en un archivo .npy
np.save('data.npy', dataConcatenada.trials)
np.save('labels.npy', dataConcatenada.labels)