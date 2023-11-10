from EEGLogger.EEGLogger import EEGLogger, setupBoard

from SignalProcessor.Filter import Filter
# from SignalProcessor.RavelTransformer import RavelTransformer
# from SignalProcessor.FeatureExtractor import FeatureExtractor
from ArduinoCommunication.ArduinoCommunication import ArduinoCommunication

import json
import os
import time
import random
import logging

import numpy as np
import pandas as pd
import pickle

import sys
from PyQt5.QtCore import QTimer#, QThread, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QTime, QDate, QDateTime
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from GUIModule.IndicatorAPP import IndicatorAPP
from GUIModule.ConfigAPP import ConfigAPP
from GUIModule.SupervisionAPP import SupervisionAPP
from GUIModule.InfoAPP import InfoAPP

from sklearn.pipeline import Pipeline

class Core(QMainWindow):
    """Esta clase es la clase principal del sistema.
    Clase para manejar el bloque de procesamiento de los datos (filtrado, extracción de característica y clasificación),
    las GUI y el bloque de comunicación con el dispositivo de control. 
    Esta clase usará las clases EEGLogger, Filter, FeatureExtractor y Classifier para procesar la señal de EEG.
    NOTA: Queda pendiente ver la implementación de al menos un hilo para controlar el Filter, CSPMultcilass, FeatureExtractor,
    RavelTransformer y Classifier. Este hilo deberá ser el que controle los tiempos de de inicio/inter-trial, el tiempo
    del cue y el de descanso. Además, deberá ser el encargado de enviar los comandos al dispositivo de control (o bien,
    se puede hacer otro para el control de la comunicación con el dispositivo.)
    

    FLUJO BÁSICO DEL PROGRAMA
    Si bien se pueden tener tres tipos de sesiones diferentes, el flujo básico del programa es el siguiente:
    1. Se inicia el contrsuctor. Dentro del mismo se configuran parámetros importantes y se inicializan las clases
    ConfigAPP, IndicatorAPP y SupervisionAPP. Además, se configuran los timers para controlar el inicio de la sesión,
    el final de la sesión y el tiempo de cada trial. Finalmente se llama showGUIAPPs() para mostrar las GUI el cual
    inica las app y el timer configAppTimer.
    2. Una vez que se inicia la sesión, se cierra la app de configuración y se llama a la función self.start() para iniciar
    el timer self.iniSesionTimer el cual controla self.startSesion() quien inicia la comunicación con la placa y dependiendo
    el tipo de sesión iniciará el timer self.trainingEEGThreadTimer o self.feedbackThreadTimer.
    3. Estos timers controla los procesos dependiendo del tipo de sesión. Pero de manera general se encargan de controlar
    el tiempo de cada trial y de cada fase del trial. Por ejemplo, en el caso de la sesión de entrenamiento, el timer
    self.trainingEEGThreadTimer controla el tiempo de cada trial y de cada fase del trial. En la fase de inicio, se muestra la cruz
    y se espera un tiempo aleatorio entre self.startingTimes[0] y self.startingTimes[1]. Luego, se pasa a la fase de cue
    y finalmente a la fase de finalización del trial. En esta última fase se guardan los datos de EEG y se incrementa el
    número de trial. Este proceso se repite hasta que se alcanza el último trial de la sesión.

    """
    def __init__(self, configParameters, configAPP, indicatorAPP, supervisionAPP):
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
            -lenToClassify (float): Tiempo a usar para clasificar la señal de EEG.
            -lenForClassifier (float): Tiempo total de EEG para alimentar el clasificador.
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
            -training_events_file (str): Ruta al archivo txt con los eventos registrados durante las sesiones
            -classifierFile (str): Ruta al archivo pickle con el clasificador. IMPORTANTE: Se supone que este archivo ya fue generado con la sesión
            de entrenamiento y será usado durante las sesiones de feedback y online.
        Un trial es la suma de startingTimes + cueDuration + finishDuration

        - indicatorAPP (QWidget): Objeto de la clase Entrenamiento. Se usa para enviar señales a la GUI.
        - supervisionAPP (QWidget): Objeto de la clase Supervision. Se usa para supervisar eventos, señal de EEG entre otros.
        
        NOTA: Definir qué parámetros se necesitan inicar dentro del constructor."""

        super().__init__() #Inicializamos la clase padre

        self.configAPP = configAPP
        self.indicatorAPP = indicatorAPP
        self.__supervisionAPPClass = supervisionAPP

        #Parámetros generales para la sesións
        self.configParameters = configParameters

        self.typeSesion = configParameters["typeSesion"]
        self.cueType = configParameters["cueType"]
        self.ntrials = configParameters["ntrials"]
        self.classes = configParameters["classes"]
        self.clasesNames = configParameters["clasesNames"]
        self.startingTimes = configParameters["startingTimes"]
        self.cueDuration = configParameters["cueDuration"]
        self.finishDuration = configParameters["finishDuration"]
        self.lenToClassify = configParameters["lenToClassify"]
        self.lenForClassifier = configParameters["lenForClassifier"]
        self.subjectName = configParameters["subjectName"]
        self.sesionNumber = configParameters["sesionNumber"]

        #Parámetros para inicar la placa openbci
        self.boardParams = configParameters["boardParams"]
        self.channels = self.boardParams["channels"]
        self.serialPort = self.boardParams["serialPort"]
        self.boardName = self.boardParams["boardName"]

        #parámetros del filtro
        self.filterParameters = configParameters["filterParameters"]
        #Chequeamos el tipo de placa para corregir el sample rate
        if self.boardParams["boardName"] == "cyton":
            self.filterParameters["sample_rate"] = 250.
        elif self.boardParams["boardName"] == "ganglion":
            self.filterParameters["sample_rate"] = 200.
        elif self.boardParams["boardName"] == "synthetic":
            self.filterParameters["sample_rate"] = 250.

        self.sample_rate = self.filterParameters["sample_rate"]

        ## Archivo de eventos de una sesión de entrenamiento
        self.training_events_file = configParameters["events_file"]

        ## Archivo para cargar el clasificador
        self.classifierFile = configParameters["classifierFile"]

        #archivo para cargar el pipeline
        self.pipelineFile = configParameters["pipelineFile"]

        self.__trialPhase = 0 #0: Inicio, 1: Cue, 2: Finalización
        self.__trialNumber = 0 #Número de trial actual
        self.__startingTime = self.startingTimes[1]
        self.rootFolder = "data/"

        self.session_started = False #Flag para indicar si se inició la sesión

        #Configuramos timers del Core
        """
        Funcionamiento QTimer
        https://stackoverflow.com/questions/42279360/does-a-qtimer-object-run-in-a-separate-thread-what-is-its-mechanism
        """

        #timer para controlar el inicio de la sesión
        self.iniSesionTimer = QTimer()
        self.iniSesionTimer.setInterval(1000) #1 segundo1
        self.iniSesionTimer.timeout.connect(self.startSesion)

        #timer para controlar si se alcanzó el último trial y así cerrar la app
        self.checkTrialsTimer = QTimer()
        self.checkTrialsTimer.setInterval(10) #Chequeamos cada 10ms
        self.checkTrialsTimer.timeout.connect(self.checkLastTrial)

        #timer para controlar las fases de cada trial
        self.trainingEEGThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.trainingEEGThreadTimer.setInterval(int(self.startingTimes[1]*1000)) #1 milisegundo sólo para el inicio de sesión.
        self.trainingEEGThreadTimer.timeout.connect(self.trainingEEGThread)

        self.feedbackThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.feedbackThreadTimer.setInterval(int(self.startingTimes[1]*1000)) #1 milisegundo sólo para el inicio de sesión.
        self.feedbackThreadTimer.timeout.connect(self.feedbackThread)

        self.onlineThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.onlineThreadTimer.setInterval(int(self.startingTimes[1]*1000)) #1 milisegundo sólo para el inicio de sesión.
        self.onlineThreadTimer.timeout.connect(self.onlineThread)

        #timer para controlar el tiempo para clasificar el EEG
        self.classifyEEGTimer = QTimer()
        self.classifyEEGTimer.setInterval(int(self.lenToClassify*1000)) #Tiempo en milisegundos
        self.classifyEEGTimer.timeout.connect(self.classifyEEG)

        #timer para controlar la app de configuración
        self.configAppTimer = QTimer()
        self.configAppTimer.setInterval(5) #ms
        self.configAppTimer.timeout.connect(self.checkConfigApp)

        #timer para actualizar la supervisionAPP
        self.__supervisionAPPTime = 10 #ms
        self.supervisionAPPTimer = QTimer()
        self.supervisionAPPTimer.setInterval(self.__supervisionAPPTime)
        self.supervisionAPPTimer.timeout.connect(self.updateSupervisionAPP)

        self.showGUIAPPs()

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
        self.lenForClassifier = newParameters["lenForClassifier"]
        self.subjectName = newParameters["subjectName"]
        self.sesionNumber = newParameters["sesionNumber"]
        self.boardParams = newParameters["boardParams"]
        self.channels = self.boardParams["channels"]
        self.serialPort = self.boardParams["serialPort"]
        self.boardName = self.boardParams["boardName"]

        self.channels = self.boardParams["channels"]

        self.filterParameters = newParameters["filterParameters"]
        #Chequeamos el tipo de placa para corregir el sample rate
        if self.boardParams["boardName"] == "cyton":
            self.filterParameters["sample_rate"] = 250.
        elif self.boardParams["boardName"] == "ganglion":
            self.filterParameters["sample_rate"] = 200.
        elif self.boardParams["boardName"] == "synthetic":
            self.filterParameters["sample_rate"] = 250.

        self.sample_rate = self.filterParameters["sample_rate"]
        
        self.training_events_file = newParameters["events_file"]
        self.classifierFile = newParameters["classifierFile"]

        #archivo para cargar el pipeline
        self.__customPipeline = newParameters["customPipeline"]
        self.pipelineFile = newParameters["pipelineFile"]

        self.__startingTime = self.startingTimes[1]

        self.classifyEEGTimer.setInterval(int(self.lenToClassify*1000)) #Tiempo en milisegundos

        #actualizamos el diccionario
        self.configParameters = newParameters

    def saveConfigParameters(self, fileName = None):
        """Guardamos el diccionario configParameters en un archivo json"""

        if not fileName:
            with open(self.eegStoredFolder+self.eegFileName+"config.json", 'w') as f:
                json.dump(self.configParameters, f, indent = 4)
        else:
            with open(fileName, 'w') as f:
                json.dump(self.configParameters, f, indent = 4)

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
        #Si la carpeta rootFolder/self.subjectName/eegdata/self.sesionNumber no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/eegdata" + f"/sesion{str(self.sesionNumber)}"):
            os.makedirs(rootFolder + self.subjectName + "/eegdata" + f"/sesion{str(self.sesionNumber)}")
        self.eegStoredFolder = self.rootFolder + self.subjectName + "/eegdata/" + f"/sesion{str(self.sesionNumber)}/"

        #Chequeamos si el eegFileName existe. Si existe, se le agrega un número al final para no repetir
        #el nombre del archivo por defecto es self.eegFileName =  "s{self.sesionNumber}_t{self.cueType}_r1.npy"
        #Donde, s = sesión_number, ts = type_sesion, ct = cue_type, r = run_number
        self.eegFileName =  f"sn{self.sesionNumber}_ts{self.typeSesion}_ct{self.cueType}_r1.npy"
        if os.path.exists(self.eegStoredFolder + self.eegFileName):
            i = 2
            while os.path.exists(self.eegStoredFolder + self.eegFileName):
                self.eegFileName =  f"sn{self.sesionNumber}_ts{self.typeSesion}_ct{self.cueType}_r{i}.npy"
                i += 1
        
        #Cramos un archivo txt que contiene la siguiente cabecera:
        #trialNumber, classNumber, className,startingTime,cueDuration,trialTime,time-time(formateado)
        #Primero creamos el archivo y agregamos la cabecera. Lo guardamos en rootFolder/self.subjectName/eegdata/self.sesionNumber
        #con el mismo nombre que self.eegFileName pero con extensión .txt
        self.eventsFileName = self.eegStoredFolder + self.eegFileName[:-4] + "_events" + ".txt"
        eventsFile = open(self.eventsFileName, "w")
        eventsFile.write("trialNumber,classNumber,className,prediction,probabilities,startingTime,cueDuration,finishDuration,trialTime,trialTime(legible)\n")
        eventsFile.close()
        
        #Si la carpeta pipelines dentro de self.subjectName no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/pipelines"):
            os.makedirs(rootFolder + self.subjectName + "/pipelines")

    def saveEvents(self):
        """Función para almacenar los eventos de la sesión en el archivo txt self.eventsFileName
        La función almacena los siguientes eventos, self.trialNumber, self.classNumber, self.class,
        self.trialPhase, self.trialTime, self.trialStartTime, self.trialEndTime.
        Cada nuevo dato se agrega en una nueva linea. Se abre el archivo en modo append (a)"""
        
        eventsFile = open(self.eventsFileName, "a")
        
        claseActual = self.trialsSesion[self.__trialNumber]
        classNameActual = self.clasesNames[self.classes.index(claseActual)]

        #obtenemos el timestamp actual
        trialTime = time.time()
        #formateamos el timestamp actual a formato legible del tipo DD/MM/YYYY HH:MM:SS
        trialTimeLegible = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(trialTime))

        if self.typeSesion == 0:
            eventos = f"{self.__trialNumber+1},{claseActual},{classNameActual},{-1},{-1},{self.__startingTime},{self.cueDuration},{self.finishDuration},{trialTime},{trialTimeLegible}\n"
        
        elif self.typeSesion == 1:
            eventos = f"{self.__trialNumber+1},{claseActual},{classNameActual},{self.prediction[-1]},{self.probas[-1]},{self.__startingTime},{self.cueDuration},{self.finishDuration},{trialTime},{trialTimeLegible}\n"

        elif self.typeSesion == 2:
            pass

        eventsFile.write(eventos)
        eventsFile.close()

    def setEEGLogger(self, startStreaming = True):
        """Seteamos EEGLogger para lectura de EEG desde placa.
        Iniciamos streaming de EEG."""
        
        print("Seteando EEGLogger...")
        logging.info("Seteando EEGLogger...")
        board, board_id = setupBoard(boardName = self.boardName, serial_port = self.serialPort)
        self.eeglogger = EEGLogger(board, board_id)
        self.eeglogger.connectBoard()
        self.eeglogger.setStreamingChannels(self.channels)
        time.sleep(1) #esperamos 1 segundo para que se conecte la placa
        print("Iniciando streaming de EEG...")
        logging.info("Iniciando streaming de EEG...")

        channels_names = self.eeglogger.board.get_eeg_channels(board_id)

        if startStreaming:
            self.eeglogger.startStreaming()#iniciamos streaming de EEG
            print("Esperamos para estabilizar señal de EEG...")
            time.sleep(3) #Esperamos unos segundos para estabilizar la señal de EEG
            
            #iniciamos timer para actualizar grafico de EEG de la supervisionAPP
            self.supervisionAPPTimer.start()

    def setFilter(self):
        """Función para setear el filtro de EEG que usaremos en la supervisiónAPP
        - Los parámetros del filtro se obtienen a partir de self.parameters['filterParameters']"""

        lowcut = self.filterParameters['lowcut']
        highcut = self.filterParameters['highcut']
        notch_freq = self.filterParameters['notch_freq']
        notch_width = self.filterParameters['notch_width']
        sample_rate = self.filterParameters['sample_rate']
        axisToCompute = self.filterParameters['axisToCompute']

        self.filter = Filter(lowcut=lowcut, highcut=highcut, notch_freq=notch_freq, notch_width=notch_width,
                             sample_rate=sample_rate, axisToCompute = axisToCompute,
                             padlen = int(self.__supervisionAPPTime * sample_rate/1000)-1)

    def setPipeline(self, **pipelineBlocks):
        """Función para setear el pipeline para el procesamiento y clasificación de EEG.
        Parametros:
        - filename (str): nombre del archivo (pickle) donde se encuentra el pipeline guardado. Si es None
        se setea el pipeline con los parámetros dados en pipelineObject.
        - pipelineBlocks (dict): diccionario con los diferentes objetos para el pipeline.
        """
        
        #Si pipelineBlocks esta vacío, se carga el pipeline desde el archivo self.pipelineFileName
        if not pipelineBlocks: #cargamos pipeline desde archivo. ESTO ES LO RECOMENDABLE
            self.pipeline = pickle.load(open(self.pipelineFile, "rb")) #cargamos el pipeline

        #Si pipelineBlocks no esta vacío, se setea el pipeline con los parámetros dados en pipelineObject
        else:
            self.pipeline = Pipeline([(step, pipelineBlocks[step]) for step in pipelineBlocks.keys()])

    def makeAndMixTrials(self):
        """Clase para generar los trials de la sesión. La cantidad total de trials
        está dada por [ntrials * len(self.classes)].
        Se genera un numpy array de valores correspondiente a cada clase y se mezclan.
        
        Retorna:
            -trialsSesion (list): numpyarray con los trials de la sesión."""

        self.trialsSesion = np.array([[i] * self.ntrials for i in self.classes]).ravel()
        random.shuffle(self.trialsSesion)

    def checkLastTrial(self):
        """Función para chequear si se alcanzó el último trial de la sesión.
        Se compara el número de trial actual con el número de trials totales dado en self.trialsSesion"""
        if self.__trialNumber == len(self.trialsSesion):
            print("Último trial alcanzado")
            print("Sesión finalizada")
            logging.info("Se alcanzó el último trial de la sesión")
            self.checkTrialsTimer.stop()
            self.trainingEEGThreadTimer.stop()
            self.feedbackThreadTimer.stop()
            self.eeglogger.stopBoard()
            self.supervisionAPPTimer.stop()
            self.closeApp()
        else:
            pass

    def trainingEEGThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Sólo se almacena trozos de EEG correspondientes a la suma de startTrainingTime y cueDuration.
        """

        if self.__trialPhase == 0:
            print(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            logging.info(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            self.indicatorAPP.showCruz(True) #mostramos la cruz
            self.indicatorAPP.update_order("Fijar la mirada en la cruz...")
            #Generamos un número aleatorio entre self.startingTimes[0] y self.startingTimes[1], redondeado a 1 decimal
            startingTime = round(random.uniform(self.startingTimes[0], self.startingTimes[1]), 1)
            self.__startingTime = startingTime
            startingTime = int(startingTime * 1000) #lo pasamos a milisegundos
            self.__trialPhase = 1 # pasamos a la siguiente fase -> CUE
            self.trainingEEGThreadTimer.setInterval(startingTime) #esperamos el tiempo aleatorio

        elif self.__trialPhase == 1:
            self.indicatorAPP.showCruz(False) #desactivamos la cruz
            logging.info("Iniciamos fase cue del trial")
            claseActual = self.trialsSesion[self.__trialNumber]
            classNameActual = self.clasesNames[self.classes.index(claseActual)]
            self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
                                              background = "rgb(38,38,38)", font_color = "white")
            self.__trialPhase = 2 # la siguiente fase es la de finalización del trial
            self.trainingEEGThreadTimer.setInterval(int(self.cueDuration * 1000))

            ##genero un array de 5 elementos con números elatotios entre 0 y 1
            ##Estos números representan la probabilidad de que se muestre cada barra
            ##La suma de estos números debe ser 1
            ##Esto se hace para que la barra se mueva de manera aleatoria
            ##Si se quiere que la barra se mueva de manera lineal, se puede usar un array de 5 elementos
            probas = np.random.rand(5)
            probas = probas/np.sum(probas)
            self.supervisionAPP.update_propbars(probas)

        elif self.__trialPhase == 2:
            logging.info("Iniciamos fase de finalización del trial")
            self.indicatorAPP.update_order("Fin de tarea...")
            self.__trialPhase = -1 #Fase para guardar datos de EEG
            self.trainingEEGThreadTimer.setInterval(int(self.finishDuration * 1000))

        else:
            #Al finalizar el trial, guardamos los datos de EEG
            logging.info("Guardando datos de EEG")
            newData = self.eeglogger.getData(self.__startingTime + self.cueDuration + self.finishDuration, removeDataFromBuffer=False)
            self.eeglogger.saveData(newData[self.channels], fileName = self.eegFileName, path = self.eegStoredFolder, append=True)
            self.saveEvents() #guardamos los eventos de la sesión
            self.__trialPhase = 0 #volvemos a la fase inicial del trial
            self.supervisionAPP.reset_timeBar = True
            self.__trialNumber += 1 #incrementamos el número de trial
            self.trainingEEGThreadTimer.setInterval(1)

    def feedbackThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Sólo se almacena trozos de EEG correspondientes a la suma de startTrainingTime y cueDuration.
        """

        if self.__trialPhase == 0:
            print(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            logging.info(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            self.indicatorAPP.showCruz(True) #mostramos la cruz
            self.indicatorAPP.update_order("Fijar la mirada en la cruz...")
            #Generamos un número aleatorio entre self.startingTimes[0] y self.startingTimes[1], redondeado a 1 decimal
            startingTime = round(random.uniform(self.startingTimes[0], self.startingTimes[1]), 1)
            self.__startingTime = startingTime
            startingTime = int(startingTime * 1000) #lo pasamos a milisegundos
            self.__trialPhase = 1 # pasamos a la siguiente fase -> CUE
            self.feedbackThreadTimer.setInterval(startingTime) #esperamos el tiempo aleatorio

        elif self.__trialPhase == 1:
            logging.info("Iniciamos fase cue del trial")
            self.indicatorAPP.showCruz(False)
            claseActual = self.trialsSesion[self.__trialNumber]
            classNameActual = self.clasesNames[self.classes.index(claseActual)]
            self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
                                              background = "rgb(38,38,38)", font_color = "white")
            self.indicatorAPP.showBar(True)
            self.indicatorAPP.actualizar_barra(0) #iniciamos la barra en 0%
            self.__trialPhase = 2 # la siguiente fase es la de finalización del trial

            # Tomo los datos de EEG de la duración del cue y los guardo en self._dataToClasify 
            # Los datos dentro de self._dataToClasify se van actualizando en cada entrada a la función classifyEEG
            # self._dataToClasify = self.eeglogger.getData(self.cueDuration, removeDataFromBuffer=False)[self.channels]
            self._dataToClasify = self.eeglogger.getData(self.lenForClassifier, removeDataFromBuffer=False)[self.channels]
            self.classifyEEGTimer.start() #inicio el timer para clasificar el EEG
            self.feedbackThreadTimer.setInterval(int((self.cueDuration + self.lenToClassify*0.05) * 1000))
            #La suma de self.cueDuration + self.lenToClassify*0.05 es para darle un pequeño margen de tiempo

        elif self.__trialPhase == 2:
            self.classifyEEGTimer.stop() #detenemos el timer de clasificación
            self.indicatorAPP.showBar(False)
            logging.info("Iniciamos fase de finalización del trial")
            self.indicatorAPP.update_order("Fin de tarea...")
            self.__trialPhase = -1 #volvemos a la fase inicial del trial
            self.feedbackThreadTimer.setInterval(int(self.finishDuration * 1000))

        else:
            #Al finalizar el trial, guardamos los datos de EEG
            logging.info("Guardando datos de EEG")
            newData = self.eeglogger.getData(self.__startingTime + self.cueDuration + self.finishDuration, removeDataFromBuffer=False)
            self.eeglogger.saveData(newData[self.channels], fileName = self.eegFileName, path = self.eegStoredFolder, append=True)
            self.saveEvents() #guardamos los eventos de la sesión
            self.__trialPhase = 0 #volvemos a la fase inicial del trial
            self.__trialNumber += 1 #incrementamos el número de trial
            self.feedbackThreadTimer.setInterval(1)

    def onlineThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Sólo se almacena trozos de EEG correspondientes a la suma de startTrainingTime y cueDuration.
        """

        # logging.info("Iniciamos fase cue del trial")
        # self.indicatorAPP.showCruz(False)
        # claseActual = self.trialsSesion[self.__trialNumber]
        # classNameActual = self.clasesNames[self.classes.index(claseActual)]
        # self.indicatorAPP.update_order(f"{classNameActual}", fontsize = 46,
        #                                 background = "rgb(38,38,38)", font_color = "white")
        # self.indicatorAPP.showBar(True)
        # self.indicatorAPP.actualizar_barra(0) #iniciamos la barra en 0%
        # self.__trialPhase = 2 # la siguiente fase es la de finalización del trial

        # # Tomo los datos de EEG de la duración del cue y los guardo en self._dataToClasify 
        # # Los datos dentro de self._dataToClasify se van actualizando en cada entrada a la función classifyEEG
        # # self._dataToClasify = self.eeglogger.getData(self.cueDuration, removeDataFromBuffer=False)[self.channels]
        # self._dataToClasify = self.eeglogger.getData(self.lenForClassifier, removeDataFromBuffer=False)[self.channels]
        # self.classifyEEGTimer.start() #inicio el timer para clasificar el EEG

        self.onlineThreadTimer.setInterval(int((self.cueDuration + self.lenToClassify*0.05) * 1000))
        #La suma de self.cueDuration + self.lenToClassify*0.05 es para darle un pequeño margen de tiempo

    def showGUIAPPs(self):
        """Función para configurar la sesión de entrenamiento usando self.confiAPP.
        """
        self.indicatorAPP.show() #mostramos la APP
        self.indicatorAPP.update_order("Configurando la sesión...")
        self.configAPP.show() #mostramos la APP
        self.configAppTimer.start()

    def checkConfigApp(self):
        """Función para comprobar si la configuración de la sesión ha finalizado."""
        if not self.configAPP.is_open:
            print("CONFIG APP CERRADA")
            newParameters = self.configAPP.getParameters()
            self.updateParameters(newParameters)
            self.configAppTimer.stop()
            self.start() #iniciamos la sesión

    def updateSupervisionAPP(self):
        """Función para actualizar la APP de supervisión."""
        ##Actualizamos gráficas de EEG y FFTgit

        #obtenemos los datos de EEG
        data = self.eeglogger.getData(self.__supervisionAPPTime/1000, removeDataFromBuffer = False)[self.channels]

        self.supervisionAPP.update_plots(data)

        if self.session_started:
            #actualizamos información de la sesión
            self.supervisionAPP.update_info(self.typeSesion,
                                            self.__startingTime + self.cueDuration + self.finishDuration,
                                            self.__trialPhase,
                                            self.__trialNumber,
                                            len(self.trialsSesion))
            
            self.supervisionAPP.update_timebar(self.__startingTime + self.cueDuration + self.finishDuration,
                                               self.__supervisionAPPTime/1000, self.__trialPhase)

    def classifyEEG(self):
        """Función para clasificar EEG
        La función se llama cada vez que se activa el timer self.classifyEEGTimer. La duración
        del timer esta dada por self.classifyEEGTimer.setInterval().

        Se obtiene un nuevo trozo de EEG de longitud self.lenToClassify segundos, se añade al
        buffer de datos a clasificar y se clasifica. El resultado de la clasificación se almacena
        en self.prediction.

        Por cada entrada a la función, se elimina el primer trozo de datos del buffer de datos a
        clasificar y se añade el nuevo trozo de datos. La idea es actualizar los datos mientras la persona ejecuta
        la tarea.
        """
        newData = self.eeglogger.getData(self.lenToClassify, removeDataFromBuffer = False)[self.channels]
        samplesToRemove = int(self.lenToClassify*self.sample_rate) #muestras a eliminar del buffer interno de datos

        self._dataToClasify = np.roll(self._dataToClasify, -samplesToRemove, axis=1)
        self._dataToClasify[:, -samplesToRemove:] = newData

        channels, samples = self._dataToClasify.shape

        #camibamos la forma de los datos para que sea compatible con el modelo
        trialToPredict = self._dataToClasify.reshape(1,channels,samples)
        self.prediction = self.pipeline.predict(trialToPredict) #aplicamos data al pipeline
        self.probas = self.pipeline.predict_proba(trialToPredict) #obtenemos las probabilidades de cada clase

        #actualizo barras de probabilidad en supervision app
        self.supervisionAPP.update_propbars(self.probas[0])

        ## nos quedamos con la probabilida de la clase actual
        probaClaseActual = self.probas[0][self.classes.index(self.trialsSesion[self.__trialNumber])]
        self.indicatorAPP.actualizar_barra(probaClaseActual) #actualizamos la barra de probabilidad
        
    def start(self):
        """Método para iniciar la sesión"""
        print(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")
        logging.info(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")
        logging.info(f"Iniciando APP de Supervisión")

        self.supervisionAPP = self.__supervisionAPPClass([str(clase) for clase in self.classes], self.channels)
        self.supervisionAPP.show() #mostramos la APP de supervisión

        if self.typeSesion == 0:
            self.indicatorAPP.update_order("Iniciando sesión de entrenamiento") #actualizamos app
        
        if self.typeSesion == 1:
            self.indicatorAPP.update_order("Iniciando sesión de feedback") #actualizamos app

        self.iniSesionTimer.start()

    def sanityChecks(self):
        """Método chequear diferentes parámetros antes de iniciar la sesión para no crashear
        durante la sesión o de Entrenamiento, de Calibración u Online cuando estas están en marcha."""

        print("Iniciando Sanity Check...")
        logging.info("Iniciando Sanity Check...")

        training_events = pd.read_csv(self.training_events_file, sep = ",")
        # trained_classesNames = training_events["className"].unique()
        trained_classesValues = np.sort(training_events["classNumber"].unique())
        train_samples = int(training_events["cueDuration"].unique()*self.sample_rate)

        n_channels = len(self.channels)
        classify_samples = int(self.sample_rate * self.lenForClassifier)

        ## Chequeos
        ## Chequeamos que self.classes y self.clasesNames tengan la misma cantidad de elementos
        if len(self.classes) != len(self.clasesNames):
            self.closeApp()
            logging.error("La cantidad de clases y la cantidad de nombres de clases deben ser iguales")
            raise Exception("La cantidad de clases y la cantidad de nombres de clases deben ser iguales")
        
        ## chequeamos que no se repitan nombres en self.clasesNames
        if len(self.clasesNames) != len(set(self.clasesNames)):
            self.closeApp()
            logging.error("Hay nombres de clases repetidos")
            raise Exception("Hay nombres de clases repetidos")
        
        ## chequeamos que nos e repitan valores en self.classes
        if len(self.classes) != len(set(self.classes)):
            self.closeApp()
            logging.error("Hay valores de clases repetidos")
            raise Exception("Hay valores de clases repetidos")
        
        ## Chequeamos que la duración del trial sea igual al utilizado para entrenar el clasificador
        if train_samples != classify_samples:
            self.closeApp()
            logging.error("La duración del trial a clasificar debe ser igual al utilizado para entrenar el clasificador")
            raise Exception("La duración del trial a clasificar debe ser igual al utilizado para entrenar el clasificador")

        ## Chequeamos que los trained_classesValues estén presentes dentro de self.classes
        if not np.any(np.isin(trained_classesValues, self.classes)):
            ## me quedo con los valores que no están en self.classes
            values_not_in_classes = trained_classesValues[~np.isin(trained_classesValues, self.classes)]
            self.closeApp()
            logging.error("Hay una o más clases a utilizar que no se usaron durante en la sesión de entrenamiento", values_not_in_classes)
            raise Exception("Hay una o más clases a utilizar que no se usaron durante en la sesión de entrenamiento", values_not_in_classes)
 
        ## generamos un numpy array con valores enteros igual a 1. El shape es de la forma [1, n_channels, classify_samples]
        ## Este array representa un trial de EEG
        trial = np.ones((1, n_channels, classify_samples), dtype=np.int8)

        try:
            self.pipeline.predict(trial)
        except ValueError as e:
            self.closeApp()
            print(e)
            mensaje = "Compruebe que la cantidad de canales a usar se correspondan con la cantidad de canales usada durante el entrenamiento del clasificador"
            logging.error(mensaje)
            raise Exception(mensaje)
        
        ##chequeamos si self.pipeline posee el método predict_proba
        if not hasattr(self.pipeline, "predict_proba"):
            self.closeApp()
            logging.error("El pipeline no posee el método predict_proba")
            raise Exception("El pipeline no posee el método predict_proba")
        else: #si lo posee, chequeamos que la cantidad de probabilidades retornada sea igual a la cantidad de clases
            probas = self.pipeline.predict_proba(trial)
            if len(probas[0]) != len(self.classes):
                self.closeApp()
                mensaje = "La cantidad de probabilidades retornada por el pipeline es diferente a la cantidad de clases que se intenta clasificar. \nLa cantidad y el tipo de clases a clasificar debe corresponderse con la usada durante el entrenamiento del clasificador"
                logging.error(mensaje)
                raise Exception(mensaje)
            
        logging.info("Sanity Check finalizado. Todo OK")
        print("Sanity Check finalizado. Todo OK")

    def startSesion(self):
        """Método para iniciar timers del Core, además
        se configuran las carpetas de almacenamiento y se guarda el archivo de configuración de la sesión.
        Se setea el setEEGLogger para comunicación con la placa.
        """
        self.iniSesionTimer.stop() #detenemos timer de inicio de sesión

        self.setFolders(rootFolder = self.rootFolder) #configuramos las carpetas de almacenamiento
        self.saveConfigParameters(self.eegStoredFolder+self.eegFileName[:-4]+"_config.json") #guardamos los parámetros de configuración
        
        self.setEEGLogger() #seteamos EEGLogger
        self.makeAndMixTrials() #generamos y mezclamos los trials de la sesión
        self.checkTrialsTimer.start()

        if self.typeSesion == 0:
            print("Inicio de sesión de entrenamiento")
            logging.info("Inicio de sesión de entrenamiento")
            self.trainingEEGThreadTimer.start() #iniciamos timer para controlar hilo entrenamiento
            self.session_started = True
            
        elif self.typeSesion == 1:
            print("Inicio de sesión de Feedback")
            logging.info("Inicio de sesión de Feedback")
            self.setPipeline() #seteamos el pipeline
            self.sanityChecks() ## sanity check
            self.feedbackThreadTimer.start() #iniciamos timer para controlar hilo calibración
            self.session_started = True

        elif self.typeSesion == 2:
            print("Inicio de sesión Online")
            ##Cerramos indicatorAPP ya que no se usa en modo Online
            self.indicatorAPP.close()

            ##usamos try/except para chequear si tenemos comunicación con arduino
            try:
                ## inicializamos la clase ArduinoCommunication
                self.arduino = ArduinoCommunication(port = self.serialPort)
            except Exception as e:
                print(e)
                logging.error(e)
                print("No se pudo establecer comunicación con arduino")
                self.closeApp()
                raise Exception("No se pudo establecer comunicación con arduino")
            
            time.sleep(1) #esperamos 1 segundo

            ##chequeamos que tenemos comunicación con arduino. Sólo nos comunicamos con arduino en la sesión online
            self.arduino.iniSesion()
            
            # self.setPipeline() #seteamos el pipeline5
            # self.sanityChecks() ## sanity check
            self.onlineThreadTimer.start() #iniciamos timer para controlar hilo calibración
            # self.session_started = True
            pass

    def closeApp(self):
        print("Cerrando aplicación...")
        self.indicatorAPP.close()
        self.supervisionAPP.close()
        self.close()

if __name__ == "__main__":

    debbuging = False
    if debbuging:
        logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    ##cargamos los parametros desde el archivo config.json
    with open("config.json", "r") as f:
        parameters = json.load(f)

    core = Core(parameters, ConfigAPP("config.json", InfoAPP), IndicatorAPP(), SupervisionAPP)

    sys.exit(app.exec_())