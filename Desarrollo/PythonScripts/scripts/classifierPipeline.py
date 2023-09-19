"""Script para generar un pipeline para evaluar diferentes combinaciones de hiperparámetros
y obtener el mejor modelo para clasificar las señales de EGG.

Script para resolver ISSUE #23
"""

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


### ********** Cargamos los datos **********
eventosFile = "data\sujeto_6\eegdata\sesion1\sn1_ts0_ct0_r1_events.txt"
file = "data\sujeto_6\eegdata\sesion1\sn1_ts0_ct0_r1.npy"
rawEEG_1 = np.load(file)
eventos_1 = pd.read_csv(eventosFile, sep = ",")

eventosFile = "data\sujeto_6\eegdata\sesion2\sn2_ts0_ct0_r1_events.txt"
file = "data\sujeto_6\eegdata\sesion2\sn2_ts0_ct0_r1.npy"
rawEEG_2 = np.load(file)
eventos_2 = pd.read_csv(eventosFile, sep = ",")

#Creamos objetos para manejar los trials
th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])
th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])

dataConcatenada = Concatenate([th_1,th_2])#concatenamos datos

channelsSelected = [0,1,2,3,6,7]

trials = dataConcatenada.trials

#me quedo con channelsSelected
trials = trials[:,channelsSelected,:]
labels = dataConcatenada.labels
classesName, labelsNames = dataConcatenada.classesName

##filtramos los trials para las clases que nos interesan
trials = trials[np.where((labels == 1) | (labels == 2) | (labels == 2) | (labels == 2) | (labels == 2))]
labels = labels[np.where((labels == 1) | (labels == 2) | (labels == 2) | (labels == 2) | (labels == 2))]

### ********** Separamos los datos en train, validation y test **********
eeg_train, eeg_test, labels_train, labels_test = train_test_split(trials, labels, test_size=0.1, stratify=labels, random_state=42)
# eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_trainBig, labels_trainBig, test_size=0.2, stratify=labels_trainBig, random_state=42)
### ********** Instanciamos los diferentes objetos que usaremos en el pipeline**********

fm = 250. #frecuencia de muestreo
filter = Filter(lowcut=8, highcut=18, notch_freq=50.0, notch_width=2, sample_rate=fm, axisToCompute=2, padlen=None, order=4)
#Creamos un CSPMulticlass - Método ovo (one vs one)
cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = len(np.unique(labels)), reg = 0.01)
featureExtractor = FeatureExtractor(method = "welch", sample_rate = fm, axisToCompute=2, band_values=[8,12])
ravelTransformer = RavelTransformer()

#Instanciamos un LDA
lda = LDA() #instanciamos el clasificador LDA

### ********** Creamos el pipeline para LDA **********

pipeline_lda = Pipeline([
    ('pasabanda', filter),
    ('cspmulticlase', cspmulticlass),
    ('featureExtractor', featureExtractor),
    ('ravelTransformer', ravelTransformer),
    ('lda', lda)
])

### ********** Creamos la grilla de hiperparámetros **********

param_grid_lda = {
    'pasabanda__lowcut': [5,8],
    'pasabanda__highcut': [12,18],
    'pasabanda__notch_freq': [50.0],
    'cspmulticlase__n_components': [2,3],
    'cspmulticlase__method': ["ovo"],
    'cspmulticlase__n_classes': [len(np.unique(labels))],
    'cspmulticlase__reg': [0.01],
    'cspmulticlase__log': [None],
    'cspmulticlase__norm_trace': [False],
    'featureExtractor__method': ["welch"],
    'featureExtractor__sample_rate': [fm],
    'featureExtractor__band_values': [[8,18]],
    'lda__solver': ['svd'],
    'lda__shrinkage': [None],
    'lda__priors': [None],
    'lda__n_components': [None],
    'lda__store_covariance': [False],
    'lda__tol': [0.0001, 0.001],
}

#Creamos el GridSearch para el LDA
grid_lda = GridSearchCV(pipeline_lda, param_grid_lda, cv=5, n_jobs=1, verbose=1)

### ********** Entrenamos el modelo **********
grid_lda.fit(eeg_train, labels_train)
### ******************************************

print("Reporte de clasificación para el mejor clasificador (sobre conjunto de evaluación):", end="\n\n")
y_true, y_pred = labels_test, grid_lda.predict(eeg_test)
print(classification_report(y_true, y_pred), end="\n\n")

### Nos quedamos con el mejor estimador
best_lda = grid_lda.best_estimator_

grid_lda_df = pd.DataFrame(grid_lda.cv_results_)
grid_lda_df.sort_values(by=["mean_test_score"], inplace=True, ascending=False)
print(grid_lda_df.columns)
#guardamos los resultados en un csv
# grid_lda_df.to_csv("grid_lda_df.csv")

## Creamos una matriz de confusión
cm_lda = confusion_matrix(y_true, y_pred)
## Obtenemos los valores en porcentaje y los redondeamos a 2 decimales
cm_lda = np.round(cm_lda.astype('float') / cm_lda.sum(axis=1)[:, np.newaxis], decimals=2)
print(cm_lda)

## Reentrenamos el mejor estimador con todo el set de entrenamiento, 
best_lda.fit(eeg_train, labels_train)
### ********** Usamos el mejor estimador para predecir los datos de testpara SCV **********
y_true, y_pred = labels_test, best_lda.predict(eeg_test)

## obtenemos precision, recall y f1-score y los guardamos en variables
precision_lda, recall_lda, f1score_lda, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

## Obtenemos el accuracy y lo redondeamos a 2 decimales
acc_lda = accuracy_score(y_true, y_pred)
acc_lda = np.round(acc_lda, decimals=2)*100
print(f"El accuracy del mejor clasificador LDA es de {acc_lda}")


### ********** Repetimos el proceso para SCV **********
### ********** Creamos el pipeline para SVC **********

#Instanciamos un SVC
svc = SVC()

pipeline_svc = Pipeline([
    ('pasabanda', filter),
    ('cspmulticlase', cspmulticlass),
    ('featureExtractor', featureExtractor),
    ('ravelTransformer', ravelTransformer),
    ('svc', svc)
])

### ********** Creamos la grilla de hiperparámetros **********
param_grid_svc = {
    'pasabanda__lowcut': [8],
    'pasabanda__highcut': [12,18],
    'pasabanda__notch_freq': [50.0],
    'cspmulticlase__n_components': [2,3],
    'cspmulticlase__method': ["ovo"],
    'cspmulticlase__n_classes': [len(np.unique(labels))],
    'cspmulticlase__reg': [0.01],
    'cspmulticlase__log': [None],
    'cspmulticlase__norm_trace': [False],
    'featureExtractor__method': ["welch", "hilbert"],
    'featureExtractor__sample_rate': [fm],
    'featureExtractor__band_values': [[8,18]],
    'svc__C': [1.0],
    'svc__kernel': ['linear','poly','rbf'],
    'svc__degree': [3],
    'svc__gamma': ['scale'],
    'svc__coef0': [0.0],
    'svc__shrinking': [True],
    'svc__probability': [False],
    'svc__tol': [0.001],
    'svc__cache_size': [200],
    'svc__class_weight': [None],
}

#creamos la grilla
grid_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=5, n_jobs=1, verbose=1)

### ********** Entrenamos el modelo **********
grid_svc.fit(eeg_train, labels_train)
### ******************************************

print("Reporte de clasificación para el mejor clasificador (sobre conjunto de evaluación):", end="\n\n")
y_true, y_pred = labels_test, grid_svc.predict(eeg_test)
print(classification_report(y_true, y_pred), end="\n\n")

## Creamos una matriz de confusión
cm_svm = confusion_matrix(y_true, y_pred)
## Obtenemos los valores en porcentaje y los redondeamos a 2 decimales
cm_svm = np.round(cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis], decimals=2)
print(cm_svm)

### ********** Usamos el mejor estimador para predecir los datos de test para SCV **********

### Nos quedamos con el mejor estimador SVM
best_svc = grid_svc.best_estimator_

### REentrenamos el mejor estimador con todo el set de entrenamiento,
best_svc.fit(eeg_train, labels_train)

y_true, y_pred = labels_test, best_svc.predict(eeg_test)

## obtenemos precision, recall y f1-score y los guardamos en variables
precision_svm, recall_svm, f1score_svm, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

## Obtenemos el accuracy y lo redondeamos a 2 decimales
acc_svm = accuracy_score(y_true, y_pred)
acc_svm = np.round(acc_svm, decimals=2)*100
print(f"El accuracy del mejor clasificador SVM es de ***{round(acc_svm,2)}%***")

### ********** Generamos un dataframe con los resultados del LDA y del SVM **********
# El dataframe tiene el accuracy, precision, recall y f1-score para el mejor clasificador LDA y SVM

df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1-Score"])
df.loc["LDA"] = [acc_lda, precision_lda, recall_lda, f1score_lda]
df.loc["SVM"] = [acc_svm, precision_svm, recall_svm, f1score_svm]

print(df)

## Guardo el mejor clasificador LDA y el mejor SVM usando pickle
# pickle.dump(best_lda, open("best_lda_subject8.pkl", "wb"))
# pickle.dump(best_svc, open("best_svc_subject8.pkl", "wb"))
