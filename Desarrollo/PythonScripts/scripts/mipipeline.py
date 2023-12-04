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

import matplotlib.pyplot as plt
    
## Clasificadores LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

## Librerias para entrenar y evaluar el modelo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import pickle
import os

list_sujetos = dict()
sujetos = [11] #[1,2,3,6,8,10,11]
for s in sujetos:
    list_df = []
    for combination in range(1,5):
        ### ********** Cargamos los datos **********
        sujeto = f"sujeto_{s}" #4 no, 5 no
        tipoTarea = "imaginado" #imaginado
        ct = 0 if tipoTarea == "ejecutado" else 1 #0 ejecutado, 1 imaginado
        comb = combination
        r = 1

        nrows = 4#"auto" ## auto para comb 1 y 2, 3 filas x 4 columnas para comb 3.... y 5 filas x f columnas para comb4
        ncols = 3#"auto"

        baseFolder = f"data\{sujeto}"
        eventosFile = f"{baseFolder}\eegdata\sesion1\sn1_ts0_ct{ct}_r{r}_events.txt"
        file = f"{baseFolder}\eegdata\sesion1\sn1_ts0_ct{ct}_r{r}.npy"
        rawEEG_1 = np.load(file)
        eventos_1 = pd.read_csv(eventosFile, sep = ",")

        eventosFile = f"{baseFolder}\eegdata\sesion2\sn2_ts0_ct{ct}_r{r}_events.txt"
        file = f"{baseFolder}\eegdata\sesion2\sn2_ts0_ct{ct}_r{r}.npy"
        rawEEG_2 = np.load(file)
        eventos_2 = pd.read_csv(eventosFile, sep = ",")

        #Creamos objetos para manejar los trials
        th_1 = TrialsHandler(rawEEG_1, eventos_1, tinit = 0.5, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])
        th_2 = TrialsHandler(rawEEG_2, eventos_2, tinit = 0.5, tmax = 4, reject=None, sample_rate=250., trialsToRemove = [])

        dataConcatenada = Concatenate([th_1,th_2])#concatenamos datos

        dataConcatenada

        channelsSelected = [0,1,2,3,6,7]

        trials = dataConcatenada.trials

        #me quedo con channelsSelected
        trials = trials[:,channelsSelected,:]
        labels = dataConcatenada.labels
        classesName, labelsNames = dataConcatenada.classesName

        comb1 = np.where((labels == 1) | (labels == 2))
        comb2 = np.where((labels == 1) | (labels == 2) | (labels == 5))
        comb3 = np.where((labels == 1) | (labels == 2) | (labels == 4) | (labels == 5))
        comb4 = np.where((labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5))

        combs = [comb1, comb2, comb3, comb4]

        #filtramos los trials para las clases que nos interesan
        trials = trials[combs [comb-1]]
        labels = labels[combs [comb-1]]

        ### ********** Analizamos los datos **********
        ##calculo la varianza de cada canal para cada trial
        filter = Filter(lowcut=8, highcut=12, notch_freq=50.0, notch_width=2, sample_rate=250.,
                        axisToCompute=2, padlen=None, order=4)

        trials_filtered = filter.fit_transform(trials)
        se = np.std(trials_filtered, axis=2)

        ##calculo el percentil 90 de la varianza de cada canal para cada trial
        q=np.percentile(se, q=95)
        bad_trials = []
        for i in range(len(se)):
            if np.any(se[i]>q):
                bad_trials.append(i)

        ##elimino bad_trials de trials y labels
        trials = np.delete(trials, bad_trials, axis=0)
        labels = np.delete(labels, bad_trials, axis=0)

        ##contamos la cantidad de labels que nos quedan por cada clase
        unique, counts = np.unique(labels, return_counts=True)

        ##eliminamos trials y labels al azar de aquellas clases que tengan más trials que la clase con menos trials
        minimo = min(counts)
        indexes_to_delete = []
        for i in range(len(unique)):
            if counts[i] > minimo:
                diff = counts[i] - minimo
                indices = np.where(labels == unique[i])
                indices = indices[0]
                indices = np.random.choice(indices, size = diff, replace=False)
                trials = np.delete(trials, indices, axis=0)
                labels = np.delete(labels, indices, axis=0)
        
        unique, counts = np.unique(labels, return_counts=True)

        ### ********** Separamos los datos en train, validation y test **********
        eeg_train, eeg_test, labels_train, labels_test = train_test_split(trials, labels, test_size=0.2, stratify=labels, random_state=42)
        # eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_trainBig, labels_trainBig, test_size=0.2, stratify=labels_trainBig, random_state=42)
        ### ********** Instanciamos los diferentes objetos que usaremos en el pipeline**********

        fm = 250. #frecuencia de muestreo
        filter = Filter(lowcut=8, highcut=18, notch_freq=50.0, notch_width=2, 
                        sample_rate=fm, axisToCompute=2, padlen=None, order=4)
        #Creamos un CSPMulticlass - Método ovo (one vs one)
        cspmulticlass = CSPMulticlass(n_components=6, method = "ova", n_classes = len(np.unique(labels)),
                                    reg = 0.01, transform_into = "average_power")#transform_into='csp_space'
        featureExtractor = FeatureExtractor(method = "welch", sample_rate = fm, axisToCompute=2, band_values=[8,12])
        ravelTransformer = RavelTransformer()

        #Instanciamos un LDA
        lda = LDA() #instanciamos el clasificador LDA

        ### ********** Creamos el pipeline para LDA **********

        pipeline_lda = Pipeline([
            ('pasabanda', filter),
            ('cspmulticlase', cspmulticlass),
            # ('featureExtractor', featureExtractor),
            # ('ravelTransformer', ravelTransformer),
            ('lda', lda)
        ])

        pipeline_lda.fit(eeg_train, labels_train)

        transformados = pipeline_lda.transform(eeg_train)
        labels_test
        clasificaciones = pipeline_lda.predict(eeg_test)

        ### ********** Creamos la grilla de hiperparámetros **********

        param_grid_lda = {
            'pasabanda__lowcut': [8],
            'pasabanda__highcut': [12],
            'pasabanda__notch_freq': [50.0],
            'cspmulticlase__n_components': [6],
            'cspmulticlase__method': ["ova"],
            'cspmulticlase__n_classes': [len(np.unique(labels))],
            'cspmulticlase__reg': [0.01],
            'cspmulticlase__log': [None],
            'cspmulticlase__norm_trace': [False],
            # 'featureExtractor__method': ["welch"],
            # 'featureExtractor__sample_rate': [fm],
            # 'featureExtractor__band_values': [[8,18]],
            'lda__solver': ['eigen'],
            'lda__shrinkage': ["auto"],
            'lda__priors': [None],
            'lda__n_components': [None],
            'lda__store_covariance': [False],
            'lda__tol': [0.1, 0.01, 0.5],
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
        # print(grid_lda_df.columns)
        #guardamos los resultados en un csv
        # grid_lda_df.to_csv("grid_lda_df.csv")

        ## Creamos una matriz de confusión
        cm_lda = confusion_matrix(y_true, y_pred)
        ## Obtenemos los valores en porcentaje y los redondeamos a 2 decimales
        cm_lda = np.round(cm_lda.astype('float') / cm_lda.sum(axis=1)[:, np.newaxis], decimals=2)
        # print(cm_lda)

        ## Reentrenamos el mejor estimador con todo el set de entrenamiento, 
        best_lda.fit(eeg_train, labels_train)
        ### ********** Usamos el mejor estimador para predecir los datos de testpara SCV **********
        y_true, y_pred = labels_test, best_lda.predict(eeg_test)

        ## obtenemos precision, recall y f1-score y los guardamos en variables
        precision_lda, recall_lda, f1score_lda, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        ## Obtenemos el accuracy y lo redondeamos a 2 decimales
        acc_lda = accuracy_score(y_true, y_pred)
        acc_lda = np.round(acc_lda, decimals=2)*100
        # print(f"El accuracy del mejor clasificador LDA es de {acc_lda}")

        precision_lda, recall_lda, f1score_lda, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1-Score"])
        df.loc["LDA"] = [acc_lda, precision_lda, recall_lda, f1score_lda]

        list_df.append(df)
        

    ##convertimos los df dentro de list_df en un solo dataframe
    df = pd.concat(list_df, axis=0)

    ##cambio los nombres del indice de comb1 a comb4
    df.index = [f"comb{i}" for i in range(1,5)]

    ##redondeamos a dos decimales
    df = np.round(df, decimals=2)

    dfFolder = "dataframes" #carpeta donde guardaremos los dataframes
    ## chequeamos si la carpeta f"{baseFolder}\{dfFolder}" existe, si no existe la creamos
    if not os.path.exists(f"{baseFolder}\{dfFolder}"):
        os.makedirs(f"{baseFolder}\{dfFolder}")
    ## Guardamos los dataframes en archivos txt
    df.to_csv(f"{baseFolder}\{dfFolder}\\df_{tipoTarea}_comb{comb}_LDA.txt", sep="\t")

    # best_lda

    # ## Guardamos el modelo
    # modelFolder = "models" #carpeta donde guardaremos los modelos
    # pipsFolder = "pipelines" #carpeta donde guardaremos los pipelines

# best_lda

# pipsFolder = "pipelines" #carpeta donde guardaremos los pipelines
# ## chequeamos si la carpeta f"{baseFolder}\{pipsFolder}" existe, si no existe la creamos
# if not os.path.exists(f"{baseFolder}\{pipsFolder}"):
#     os.makedirs(f"{baseFolder}\{pipsFolder}")
# ## Guardamos los pipelines en archivos pickle
# pickle.dump(best_lda, open(f"{baseFolder}\{pipsFolder}\\best_lda_{tipoTarea}_comb{comb}_avgpow.pkl", "wb"))

