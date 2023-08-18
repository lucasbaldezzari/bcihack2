import numpy as np
import pandas as pd

# from TrialsHandler.TrialsHandler import TrialsHandler
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

class TrialsHandler():
    """Clase para obtener los trials a partir de raw data"""

    def __init__(self, rawEEG, eventos, tinit=1, tmax=4, reject=None, sample_rate=250., trialsToRemove=None) -> None:
        """Constructor de la clase Trials"""
        self.rawEEG = rawEEG
        self.eventos = eventos.set_index("trialNumber")
        self.tinit = tinit
        self.tmax = tmax
        self.reject = reject
        self.sample_rate = sample_rate
        self.labels = self.getLabels()
        self.trials = self.getTrials()
        self.classesName = self.getClassesName()
        if trialsToRemove is not None:
            self.removeTrials(trialsToRemove)

    def getTrials(self):
        pass

    def getClassesName(self):
        clases = self.eventos[["className", "classNumber"]]
        clases = clases.drop_duplicates()
        clases = clases.sort_values(by="classNumber")
        return clases["className"].values.tolist(), clases["classNumber"].values.tolist()

    def getLabels(self):
        labels = self.eventos["classNumber"].to_numpy()
        return labels

    def saveTrials(self, filename):
        np.save(filename, self.trials)
        print("Se han guardado los trials en {}".format(filename))

    def removeTrials(self, trialsToRemove: list):
        if not isinstance(trialsToRemove, list):
            raise TypeError("trialsToRemove debe ser una lista")
        if not all(trial in self.eventos.index for trial in trialsToRemove):
            raise ValueError("Los valores de trialsToRemove no existen cómo índices en self.eventos")
        else:
            self.eventos = self.eventos.drop(trialsToRemove)
            self.trials = np.delete(self.trials, trialsToRemove, axis=0)
            self.labels = np.delete(self.labels, trialsToRemove, axis=0)
            print("Se han removido los trials {}".format(trialsToRemove))


# Implementing the modified class now
class ModifiedTrialsHandler(TrialsHandler):
    def getTrials(self):
        """Función modificada para extraer los trials dentro de self.rawEEG, tomando 10 segundos para cada trial."""
        
        # Calculamos la cantidad de muestras que representa el tinit y tmax
        tinit_samples = int(self.tinit * self.sample_rate)
        tmax_samples = int(self.tmax * self.sample_rate)

        # Calculamos la cantidad de trials
        trials = self.eventos.shape[0]
        # Calculamos la cantidad de canales
        channels = self.rawEEG.shape[0]
        # Calculamos la cantidad total de muestras por trial
        total_samples = tmax_samples

        # Creamos un array de numpy para almacenar los trials
        trialsArray = np.zeros((trials, channels, total_samples))

        # Recorremos los trials
        for trial in self.eventos.index:
            # Calculamos la cantidad de muestras que representa el startingTime.
            startingTime_samples = int(self.eventos.loc[trial]["startingTime"] * self.sample_rate)
            # Usamos startingTime_samples como punto de inicio para extraer las muestras
            trialsArray[trial-1] = self.rawEEG[:, startingTime_samples : startingTime_samples + total_samples]

        print("Se han extraido {} trials".format(trials))
        print("Se han extraido {} canales".format(channels))
        print("Se han extraido {} muestras por trial".format(total_samples))

        return trialsArray

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

# EEGs, eventos = listar_archivos('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data', "sujeto_1")

EEGs = np.load('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data/sujeto_1/eegdata/sesion1/sn1_ts0_ct0_r1.npy')
eventos = pd.read_csv('C:/Users/Admin/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/data/sujeto_1/eegdata/sesion1/sn1_ts0_ct0_r1_events.txt', sep = ",")

trialsHand = []

# for eeg, evento in zip(EEGs, eventos):
#     trialsHand.append(ModifiedTrialsHandler(EEGs, eventos, tinit=0, tmax=10, reject=None, sample_rate=250.))
trialsHand.append(ModifiedTrialsHandler(EEGs, eventos, tinit=0, tmax=10, reject=None, sample_rate=250.))
dataConcatenada = Concatenate(trialsHand)#concatenamos datos

# filtro = Filter(highcut = 16)
# dataConcatenada.trials = filtro.transform(dataConcatenada.trials)

print(dataConcatenada.trials.shape)
print(dataConcatenada.labels.shape)

# Guardar la matriz en un archivo .npy
np.save('data.npy', dataConcatenada.trials)
np.save('labels.npy', dataConcatenada.labels)