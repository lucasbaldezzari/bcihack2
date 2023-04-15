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

import sys
from PyQt5.QtCore import QTimer#, QThread, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QTime, QDate, QDateTime
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow

class Core(QMainWindow):
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

        super().__init__()

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
        self.__sleepTime = 10 #Tiempo de espera entre cada iteración del hilo principal
        self.rootFolder = "data/"

        #Configuramos timers del Core
        """
        Funcionamiento QTimer
        https://stackoverflow.com/questions/42279360/does-a-qtimer-object-run-in-a-separate-thread-what-is-its-mechanism
        """

        #timer para controlar el inicio de la sesión
        self.iniSesionTimer = QTimer()
        self.iniSesionTimer.setInterval(3000) #3 segundos
        self.iniSesionTimer.timeout.connect(self.startSesion)

        #timer para controlar si se alcanzó el último trial y así cerrar la app
        self.checkTrialsTimer = QTimer()
        self.checkTrialsTimer.setInterval(10) #Chequeamos cada 10ms
        self.checkTrialsTimer.timeout.connect(self.checkLastTrial)

        #timer para controlar las fases de cada trial
        self.phaseTrialTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.phaseTrialTimer.setInterval(1) #1 milisegundo sólo para el inicio de sesión.
        self.phaseTrialTimer.timeout.connect(self.trainingEEGTrhead)

    def start(self):
        print(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")
        logging.info(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")
        self.iniSesionTimer.start()

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

    def saveConfigParameters(self, fileName = None):
        """Guardamos el diccionario configParameters en un archivo json"""

        if not fileName:
            with open(self.eegStoredFolder+self.eegFileName+"config.json", 'w') as f:
                json.dump(self.configParameters, f)
        else:
            with open(fileName, 'w') as f:
                json.dump(self.configParameters, f)

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

        #Carpeta para almacenar la señal de EEG
        #Si la carpeta rootFolder/self.subjectName/sesions/self.sesionNumber no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/sesions" + f"/sesion{str(self.sesionNumber)}"):
            os.makedirs(rootFolder + self.subjectName + "/sesions" + f"/sesion{str(self.sesionNumber)}")
        self.eegStoredFolder = self.rootFolder + self.subjectName + "/sesions/" + f"/sesion{str(self.sesionNumber)}/"

        #Chequeamos si el eegFileName existe. Si existe, se le agrega un número al final para no repetir
        #el nombre del archivo por defecto es eegFileName =  f"sesion_{self.sesionNumber}.{self.cueType}.npy
        self.eegFileName =  f"sesion_{self.sesionNumber}.{self.cueType}.npy"
        if os.path.exists(self.eegStoredFolder + self.eegFileName):
            i = 1
            while os.path.exists(self.eegStoredFolder + self.eegFileName):
                self.eegFileName =  f"sesion_{self.sesionNumber}.{self.cueType}_{i}.npy"
                i += 1
        
        #Cramos un archivo txt que contiene la siguiente cabecera:
        #trialNumber, class, classNumber, trialPhase, trialTime, trialStartTime, trialEndTime
        #Primero creamos el archivo y agregamos la cabecera. Lo guardamos en rootFolder/self.subjectName/sesions/self.sesionNumber
        #con el mismo nombre que self.eegFileName pero con extensión .txt
        self.eventsFileName = self.eegStoredFolder + self.eegFileName[:-4] + "_events" + ".txt"
        self.eventsFile = open(self.eventsFileName, "w")
        self.eventsFile.write("trialNumber,classNumber,className,trialTime,time-time(formateado)\n")
        self.eventsFile.close()
        
        #Si la carpeta classifiers no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/classifiers"):
            os.makedirs(rootFolder + self.subjectName + "/classifiers")

        #Si la carpeta csps dentro de self.subjectName no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/csps"):
            os.makedirs(rootFolder + self.subjectName + "/csps")

    def saveEvents(self):
        """Función para almacenar los eventos de la sesión en el archivo txt self.eventsFileName
        La función almacena los siguientes eventos, self.trialNumber, self.classNumber, self.class,
        self.trialPhase, self.trialTime, self.trialStartTime, self.trialEndTime.
        Cada nuevo dato se agrega en una nueva linea. Se abre el archivo en modo append (a)"""
        
        self.eventsFile = open(self.eventsFileName, "a")
        
        # self.classes = newParameters["classes"]
        # self.clasesNames = newParameters["clasesNames"]
        #A partir de self.classes y del sefl.__trialNumber y self.trialsSesion, obtenemos el nombre de la clase
        #y el número de la clase

        claseActual = self.trialsSesion[self.__trialNumber]
        classNameActual = self.clasesNames[self.classes.index(claseActual)]

        #obtenemos el timestamp actual
        trialTime = time.time()
        #formateamos el timestamp actual a formato legible del tipo DD/MM/YYYY HH:MM:SS
        trialTimeLegible = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(trialTime))

        eventos = f"{self.__trialNumber+1},{claseActual},{classNameActual},{trialTime}-{trialTimeLegible}\n"
        
        self.eventsFile.write(eventos)
        self.eventsFile.close()

    def setEEGLogger(self, board, board_id):
        """Seteamos EEGLogger para lectura de EEG desde placa.
        Parámetros:
         - board: objeto de la clase BoardShim
         - board_id: id de la placa"""
        
        self.eeglogger = EEGLogger(board, board_id)

    def setBlocks(self):
        """Seteamos los bloques de la BCI."""

        board, board_id = setupBoard(boardName = "synthetic", serial_port = "COM5")
        self.setEEGLogger(board, board_id) #creamos el objeto EEGLogger
        self.eeglogger.connectBoard()
        time.sleep(0.5) #500ms
        self.eeglogger.startStreaming()

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

    def getTrialNumber(self):
        """Función para obtener el número de trial actual."""
        return self.__trialNumber

    def checkLastTrial(self):
        """Función para chequear si se alcanzó el último trial de la sesión.
        Se compara el número de trial actual con el número de trials totales dado en self.trialsSesion"""
        if self.__trialNumber == len(self.trialsSesion):
            print("Sesión finalizada")
            print("Último trial alcanzado")
            self.checkTrialsTimer.stop()
            self.phaseTrialTimer.stop()
            self.eeglogger.stopBoard()
            self.closeApp()
        else:
            pass

    def trainingEEGTrhead(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento."""

        if self.__trialPhase == 0:
            print(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            logging.info(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            #Generamos un número aleatorio entre self.startingTimes[0] y self.startingTimes[1], redondeado a 1 decimal
            startingTime = round(random.uniform(self.startingTimes[0], self.startingTimes[1]), 1)
            startingTime = int(startingTime * 1000) #lo pasamos a milisegundos
            self.__trialPhase = 1 # la siguiente fase la de CUE
            self.phaseTrialTimer.setInterval(startingTime) #esperamos el tiempo aleatorio

        elif self.__trialPhase == 1:
            logging.info("Iniciamos fase cue del trial")
            self.__trialPhase = 2 # la siguiente fase es la de FINISH
            self.phaseTrialTimer.setInterval(int(self.cueDuration * 1000))

        elif self.__trialPhase == 2:
            logging.info("Iniciamos fase de finalización del trial")
            #Al finalizar la fase de CUE, guardamos los datos de EEG
            newData = self.eeglogger.getData(self.cueDuration)
            self.eeglogger.saveData(newData, fileName = self.eegFileName, path = self.eegStoredFolder, append=True)
            self.saveEvents() #guardamos los eventos de la sesión
            self.__trialPhase = 0
            self.__trialNumber += 1 #incrementamos el número de trial
            self.phaseTrialTimer.setInterval(int(self.finishDuration * 1000))

    def startSesion(self):
        """Método para iniciar timers del Core"""
        self.iniSesionTimer.stop()
        self.setFolders(rootFolder = self.rootFolder)
        self.saveConfigParameters(self.eegStoredFolder+self.eegFileName[:-4]+"config.json")
        if self.typeSesion == 0:
            self.setBlocks()
            print("Inicio de sesión de entrenamiento")
            self.makeAndMixTrials()
            self.checkTrialsTimer.start()
            self.phaseTrialTimer.start() #iniciamos timer para controlar hilo entrenamiento
            
        elif self.typeSesion == 1:
            pass

        elif self.typeSesion == 2:
            pass
        
    def closeApp(self):
        print("Cerrando aplicación...")
        self.close()

if __name__ == "__main__":

    debbuging = False
    if debbuging:
        logging.basicConfig(level=logging.DEBUG)

    #Creamos un diccionario con los parámetros de configuración iniciales
    parameters = {
        "typeSesion": 0, #0: Entrenamiento, 1: Feedback, 2: Online
        "cueType": 0, #0: Se ejecutan movimientos, 1: Se imaginan los movimientos
        "classes": [1, 2, 3, 4, 5], #Clases a clasificar
        "clasesNames": ["MI", "MD", "AM", "AP", "R"], #MI: Mano izquierda, MD: Mano derecha, AM: Ambas manos, AP: Ambos pies, R: Reposo
        "ntrials": 20, #Número de trials por clase
        "startingTimes": [1., 1.1], #Tiempos para iniciar un trial de manera aleatoria entre los extremos, en segundos
        "cueDuration": 4, #En segundos
        "finishDuration": 3, #En segundos
        "lenToClassify": 0.3, #Trozo de señal a clasificar, en segundos
        "subjectName": "eegForDummyTests",
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

    app = QApplication(sys.argv)

    core = Core(parameters)    
    core.start()

    sys.exit(app.exec_())


