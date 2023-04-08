from EEGLogger.EEGLogger import EEGLogger, setupBoard

from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.Classifier import Classifier

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import random
import logging

import numpy as np

class Core():
    """Esta clase es la clase principal del sistema.
    Clase para manejar el bloque de procesamiento de los datos (filtrado, extracción de característica y clasificación),
    las GUI y el bloque de comunicación con el dispositivo de control. 
    Esta clase usará las clases EEGLogger, Filter, FeatureExtractor y Classifier para procesar la señal de EEG.
    NOTA: Queda pendiente ver la implementación de al menos un hilo para controlar el Filter, CSPMultcilass, FeatureExtractor,
    RavelTransformer y Classifier. Este hilo deberá ser el que controle los tiempos de de inicio/inter-trial, el tiempo
    del cue y el de descanso. Además, deberá ser el encargado de enviar los comandos al dispositivo de control (o bien,
    se puede hacer otro para el control de la comunicación con el dispositivo.)
    NOTA 2: Se debe pensar en un hilo para el control de la GUI.
    """
    def __init__(self, configParameters):
        """Constructor de la clase

        - Parameters (dict): Diccionario con los parámetros a ser cargados. Los parámetros son:
            -typeSesion (int): Tipo de sesión. 0: Entrenamiento, 1: Feedback o calibración, 2: Online.
            -cueType (int): 0: se ejecuta movimiento, 1: se imaginan movimientos.
            -ntrials (int): Número de trials a ejecutar.
            -classes (list): Lista de valores enteros de las clases a clasificar.
            -clasesNames (list): Lista con los nombres de las clases a clasificar.
            -startingTimes (lista): Lista con los valores mínimo y máximo a esperar antes de iniciar un nuevo cue o tarea. 
            Estos valores se usan para generar un tiempo aleatorio entre estos valores.
            -cueDuration (float): Duración del cue en segundos.
            -finishDuration (float): Duración del tiempo de finalización en segundos.
            -lenToClassify (float): Duración de la señal a clasificar en segundos.
            -subjectName (str): Nombre del sujeto.
            -sesionNumber (int): Número de la sesión.
            -boardParams (dict): Diccionario con los parámetros de la placa. Los parámetros son:
                -boardName (str): Nombre de la placa. Puede ser cyton, ganglion o synthetic.
                -serialPort (str): Puerto serial de la placa. Puede ser del tipo /dev/ttyUSB0 o COM3 (para windows).
            -filterParameters (dict): Diccionario con los parámetros del filtro. Los parámetros son:
                l-owcut (float): Frecuencia de corte inferior del filtro pasa banda.
                -highcut (float): Frecuencia de corte superior del filtro pasa banda.
                -notch_freq (float): Frecuencia de corte del filtro notch.
                -notch_width (float): Ancho de banda del filtro notch.
                -sample_rate (float): Frecuencia de muestreo de la señal.
                -axisToCompute (int): Eje a lo largo del cual se calculará la transformada.
            -featureExtractorMethod (str): Método de extracción de características. Puede ser welch o hilbert.
            -cspFile (str): Ruta al archivo pickle con los filtros CSP. IMPORTANTE: Se supone que este archivo ya fue generado con la sesión
            de entrenamiento y será usado durante las sesiones de feedback y online.
            -classifierFile (str): Ruta al archivo pickle con el clasificador. IMPORTANTE: Se supone que este archivo ya fue generado con la sesión
            de entrenamiento y será usado durante las sesiones de feedback y online.

        Un trial es la suma de startingTimes + cueDuration + finishDuration
        
        NOTA: Definir qué parámetros se necesitan inicar dentro del constructor."""

        #Parámetros generales para la sesións
        self.typeSesion = configParameters["typeSesion"]
        self.cueType = configParameters["cueType"]
        self.ntrials = configParameters["ntrials"]
        self.classes = configParameters["classes"]
        self.clasesNames = configParameters["clasesNames"]
        self.startingTimes = configParameters["startingTimes"]
        self.cueDuration = configParameters["cueDuration"]
        self.finishDuration = configParameters["finishDuration"]
        self.lenToClassify = configParameters["lenToClassify"]
        self.subjectName = configParameters["subjectName"]
        self.sesionNumber = configParameters["sesionNumber"]

        #Parámetros para inicar la placa openbci
        self.boardParams = configParameters["boardParams"]

        #parámetros del filtro
        self.filterParameters = configParameters["filterParameters"]

        ## Archivo para cargar el CSP
        self.cspFile = configParameters["cspFile"]

        ## Archivo para cargar el clasificador
        self.classifierFile = configParameters["classifierFile"]

        self.configParameters = configParameters

        self.__trialPhase = 0 #0: Inicio, 1: Cue, 2: Finalización
        self.__trialNumber = 0 #Número de trial actual

        
        self.__lock = threading.Lock() #Lock para sincronizar los hilos

    def updateParameters(self,newParameters):
        """Actualizamos cada valor dentro del diccionario
        configParameters a partir de newParameters"""
        self.typeSesion = newParameters["typeSesion"]
        self.cueType = newParameters["cueType"]
        self.ntrials = newParameters["ntrials"]
        self.classes = newParameters["classes"]
        self.clasesNames = newParameters["clasesNames"]
        self.startingTimes = newParameters["startingTimes"]
        self.cueDuration = newParameters["cueDuration"]
        self.finishDuration = newParameters["finishDuration"]
        self.lenToClassify = newParameters["lenToClassify"]
        self.subjectName = newParameters["subjectName"]
        self.sesionNumber = newParameters["sesionNumber"]
        self.boardParams = newParameters["boardParams"]
        self.filterParameters = newParameters["filterParameters"]
        self.cspFile = newParameters["cspFile"]
        self.classifierFile = newParameters["classifierFile"]
        self.configParameters = newParameters
        
    def saveConfigParameters(self, fileName):
        """Guardamos el diccionario configParameters en un archivo json"""
        with open(fileName, 'w') as fp:
            json.dump(self.configParameters, fp)

    def setEEGLogger(self, board, board_id):
        """Seteamos EEGLogger para lectura de EEG desde placa.
        Parámetros:
         - board: objeto de la clase BoardShim
         - board_id: id de la placa"""
        
        self.eeglogger = EEGLogger(board, board_id)

    def setFilter(self):
        """Iniciamos el filtro."""
        # self.filter.start()
        pass

    def setCSP(self):
        """Iniciamos el CSPMulticlass."""
        pass

    def setFeatureExtractor(self):
        """Iniciamos el FeatureExtractor."""
        # self.featureExtractor.start()
        pass

    def setClassifier(self):
        """Iniciamos el clasificador."""
        # self.classifier.start()
        pass

    def setBlocks(self):
        """Seteamos los bloques de la BCI."""

        board, board_id = setupBoard(boardName = "synthetic", serial_port = "COM5")
        self.setEEGLogger(board, board_id) #creamos el objeto EEGLogger

        # self.setFilter() #creamos el objeto Filter
        # self.setCSP() #creamos el objeto CSPMulticlass
        # self.setFeatureExtractor() #creamos el objeto FeatureExtractor
        # self.setClassifier() #creamos el objeto Classifier

    def makeAndMixTrials(self):
        """Clase para generar los trials de la sesión. La cantidad de trials es igual a
        la cantidad de trials total esta dada por [ntrials * len(self.classes)].
        Se genera una lista de valores correspondiente a cada clase y se mezclan.
        
        Retorna:
            -trialsSesion (list): numpyarray con los trials de la sesión."""

        self.trialsSesion = np.array([[i] * self.ntrials for i in self.classes]).ravel()
        random.shuffle(self.trialsSesion)

    def setFolders(self, rootFolder = "data/"):
        """Función para chequear que existan las carpetas donde se guardarán los datos de la sesión.
        En caso de que no existan, se crean.
        
        La carpeta base es la rootFolder. Dentro de esta carpeta se crean las carpetas para cada sujeto.
        
        Se usa el nombre del sujeto para crear una subcarpeta. Dentro de esta se crean las carpetas para cada sesión."""

        #si la carpeta rootFolder no existe, se crea
        if not os.path.exists(rootFolder):
            os.makedirs(rootFolder)

        #Si la carpeta rootFolder/self.subjectName no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName):
            os.makedirs(rootFolder + self.subjectName)

        #Si la carpeta rootFolder/self.subjectName/sesions/self.sesionNumber no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/sesions" + f"/sesion{str(self.sesionNumber)}"):
            os.makedirs(rootFolder + self.subjectName + "/sesions" + f"/sesion{str(self.sesionNumber)}")

        #Si la carpeta classifiers no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/classifiers"):
            os.makedirs(rootFolder + self.subjectName + "/classifiers")

        #Si la carpeta csps dentro de self.subjectName no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/csps"):
            os.makedirs(rootFolder + self.subjectName + "/csps")

    def TrainingEEGTrhead(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento."""

        print("TrainingEEGTrhead")

        if self.__trialPhase == 0:
            print("Fase de inicio de trial")
            #Generamos un número aleatorio entre self.startingTimes[0] y self.startingTimes[1], redondeado a 1 decimal
            startingTime = round(random.uniform(self.startingTimes[0], self.startingTimes[1]), 1)
            time.sleep(startingTime) #esperamos el tiempo aleatorio
            self.__trialPhase = 1 # la siguiente fase la de CUE

        elif self.__trialPhase == 1:
            print("Fase cue del trial")
            self.__trialPhase = 2 # la siguiente fase la de FINISH
            time.sleep(self.cueDuration) #esperamos el tiempo de cue

        elif self.__trialPhase == 2:
            print("Fase de finalización del trial")
            self.__trialPhase = 0
            self.__trialNumber += 1 #incrementamos el número de trial
            time.sleep(self.finishDuration) #esperamos el tiempo de finish

    def getTrialNumber(self):
        """Función para obtener el número de trial actual."""
        return self.__trialNumber

    def checkLastTrial(self):
        """Función para chequear si se alcanzó el último trial de la sesión.
        Se compara el número de trial actual con el número de trials totales dado en self.trialsSesion"""
        if self.__trialNumber == len(self.trialsSesion):
            return True
        else:
            return False

    def threadsManager(self):
        """Función para inciar, controlar y frenar los threads de la aplicación."""
        pass

if __name__ == "__main__":

    #Creamos un diccionario con los parámetros de configuración iniciales
    parameters = {
        "typeSesion": 0, #0: Entrenamiento, 1: Feedback, 2: Online
        "cueType": 0, #0: Se ejecutan movimientos, 1: Se imaginan los movimientos
        "classes": [0, 1, 2, 3, 4], #Clases a clasificar
        "clasesNames": ["MI", "MD", "AM", "AP", "R"], #MI: Mano izquierda, MD: Mano derecha, AM: Ambas manos, AP: Ambos pies, R: Reposo
        "ntrials": 1, #Número de trials por clase
        "startingTimes": [0.1, 0.2], #Tiempos para iniciar un trial de manera aleatoria entre los extremos, en segundos
        "cueDuration": 1, #En segundos
        "finishDuration": 1, #En segundos
        "lenToClassify": 0.3, #Trozo de señal a clasificar, en segundos
        "subjectName": "subject_test",
        "sesionNumber": 1,
        "boardParams": {
            "boardName": "synthetic",
            "serialPort": "COM5"
        },
        "filterParameters": {
            "lowcut": 8.,
            "highcut": 28.,
            "notch_freq": 50.,
            "notch_width": 1,
            "sample_rate": 250,
            "axisToCompute": 1
        },
        "featureExtractorMethod": "welch",
        "cspFile": "data/subject_test/csps/dummycsp.pickle",
        "classifierFile": "data/subject_test/classifiers/dummyclassifier.pickle"
    }

    #Instanciamos un objeto Core
    core = Core(parameters)

    # core.setFolders(rootFolder="data/") #Seteamos las carpetas donde se guardarán los datos.

    ### ********************************** INICIAMOS GUI CONFIGURACIÓN ********************************** ###
    ## El primer paso es iniciar la GUI de configuración
    ## Algunos campos dentro de la GUI de configuración se llenan con los parámetros iniciales
    ## Luego de que el usuario modifica los parámetros, se actualizan los parámetros.

    ## newParameters = GUIConfiguracion.getParameters() #o algo así
    # core.updateParameters(newParameters)

    ### ********************************** SETEAMOS BLOQUES ********************************** ###
    ## Una vez que tenemos los parámetros iniciales, seteamos los bloques

    ## Seteamos EEGLogger
    board, board_id = setupBoard(boardName = "synthetic", serial_port = "COM5")
    core.setEEGLogger(board, board_id) #cramos el objeto EEGLogger

    # core.eeglogger.connectBoard() #nos conectamos a la placa
    # core.eeglogger.startStreaming()
    # newData = core.eeglogger.getData(core.cueDuration)
    # newData.shape
    # core.eeglogger.stopBoard()

    # core.setFilter()
    # core.setCSP()
    # core.setFeatureExtractor()

    ### ********************************** INICIAMOS GUI ********************************** ###
    ## Dependiendo del tipo de sesión, es la GUI que iniciaremos.
    ## Si es una sesión de entrenamiento, iniciamos hilo de entrenamiento
    if core.typeSesion == 0:
        # GUIEntrenamiento()
        # core.TrainingEEGTrhead()
        core.makeAndMixTrials()
        while not core.checkLastTrial():
            with ThreadPoolExecutor(max_workers=1) as pool:
                futuro = pool.submit(core.TrainingEEGTrhead)
                print(futuro.result())

    ## Si es una sesión de feedback, iniciamos hilo de feedback
    elif core.typeSesion == 1:
        pass
    ## Si es una sesión online, iniciamos hilo para sesión online
    elif core.typeSesion == 2:
        pass

    ##TODO


