from EEGLogger.EEGLogger import EEGLogger, setupBoard

from SignalProcessor.Filter import Filter
from SignalProcessor.RavelTransformer import RavelTransformer
from SignalProcessor.FeatureExtractor import FeatureExtractor

import json
import os
from concurrent.futures import ThreadPoolExecutor
import time
import random
import logging

import numpy as np
import pickle

import sys
from PyQt5.QtCore import QTimer#, QThread, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QTime, QDate, QDateTime
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from GUIModule.IndicatorAPP import IndicatorAPP
from GUIModule.ConfigAPP import ConfigAPP
from GUIModule.SupervisionAPP import SupervisionAPP

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
    NOTA 2: Se debe pensar en un hilo para el control de la GUI.
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

        - indicatorAPP (QWidget): Objeto de la clase Entrenamiento. Se usa para enviar señales a la GUI.
        - supervisionAPP (QWidget): Objeto de la clase Supervision. Se usa para supervisar eventos, señal de EEG entre otros.
        
        NOTA: Definir qué parámetros se necesitan inicar dentro del constructor."""

        super().__init__() #Inicializamos la clase padre

        self.configAPP = configAPP
        self.indicatorAPP = indicatorAPP
        self.supervisionAPP = supervisionAPP

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
        self.subjectName = configParameters["subjectName"]
        self.sesionNumber = configParameters["sesionNumber"]

        #Parámetros para inicar la placa openbci
        self.boardParams = configParameters["boardParams"]
        self.channels = self.boardParams["channels"]

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

        ## Archivo para cargar el CSP
        self.cspFile = configParameters["cspFile"]

        ## Archivo para cargar el clasificador
        self.classifierFile = configParameters["classifierFile"]

        #archivo para cargar el pipeline
        self.__customPipeline = configParameters["customPipeline"]
        self.pipelineFile = configParameters["pipelineFile"]

        self.__trialPhase = 0 #0: Inicio, 1: Cue, 2: Finalización
        self.__trialNumber = 0 #Número de trial actual
        self.__startingTime = self.startingTimes[1]
        self.rootFolder = "data/"

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
        self.eegThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.eegThreadTimer.setInterval(1) #1 milisegundo sólo para el inicio de sesión.
        self.eegThreadTimer.timeout.connect(self.trainingEEGThread)

        self.feedbackThreadTimer = QTimer() #Timer para control de tiempo de las fases de trials
        self.feedbackThreadTimer.setInterval(1) #1 milisegundo sólo para el inicio de sesión.
        self.feedbackThreadTimer.timeout.connect(self.feedbackThread)

        #timer para controlar el tiempo para clasificar el EEG
        self.classifyEEGTimer = QTimer()
        self.classifyEEGTimer.setInterval(int(self.lenToClassify*1000)) #Tiempo en milisegundos
        self.classifyEEGTimer.timeout.connect(self.classifyEEG)

        #timer para controlar la app de configuración
        self.configAppTimer = QTimer()
        self.configAppTimer.setInterval(1) #1 ms
        self.configAppTimer.timeout.connect(self.checkConfigApp)

        #timer para actualizar la supervisionAPP
        self.supervisionAPPTimer = QTimer()
        self.supervisionAPPTimer.setInterval(10) #10 ms
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
        self.subjectName = newParameters["subjectName"]
        self.sesionNumber = newParameters["sesionNumber"]
        self.boardParams = newParameters["boardParams"]
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
        
        self.cspFile = newParameters["cspFile"]
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
        
        #Si la carpeta classifiers no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/classifiers"):
            os.makedirs(rootFolder + self.subjectName + "/classifiers")

        #Si la carpeta csps dentro de self.subjectName no existe, se crea
        if not os.path.exists(rootFolder + self.subjectName + "/csps"):
            os.makedirs(rootFolder + self.subjectName + "/csps")

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
        board, board_id = setupBoard(boardName = "synthetic", serial_port = "COM5")
        self.eeglogger = EEGLogger(board, board_id)
        self.eeglogger.connectBoard()
        time.sleep(1) #esperamos 1 segundo para que se conecte la placa
        print("Iniciando streaming de EEG...")
        logging.info("Iniciando streaming de EEG...")
        if startStreaming:
            self.eeglogger.startStreaming()#iniciamos streaming de EEG
            print("Esperamos para estabilizar señal de EEG...")
            time.sleep(3) #Esperamos unos segundos para estabilizar la señal de EEG

    def setFilter(self):
        """Función para setear el filtro de EEG.
        - Los parámetros del filtro se obtienen a partir de self.parameters['filterParameters']"""

        lowcut = self.filterParameters['lowcut']
        highcut = self.filterParameters['highcut']
        notch_freq = self.filterParameters['notch_freq']
        notch_width = self.filterParameters['notch_width']
        sample_rate = self.filterParameters['sample_rate']
        axisToCompute = self.filterParameters['axisToCompute']

        self.filter = Filter(lowcut=lowcut, highcut=highcut, notch_freq=notch_freq, notch_width=notch_width,
                             sample_rate=sample_rate, axisToCompute = axisToCompute)
        
    def setPipeline(self, **pipelineBlocks):
        """Función para setear el pipeline para el procesamiento y clasificación de EEG.
        Parametros:
        - filename (str): nombre del archivo (pickle) donde se encuentra el pipeline guardado. Si es None
        se setea el pipeline con los parámetros dados en pipelineObject.
        - pipelineBlocks (dict): diccionario con los diferentes objetos para el pipeline.
        """
        
        #Si pipelineBlocks esta vacío, se carga el pipeline desde el archivo self.pipelineFileName
        if not pipelineBlocks:
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
            self.eegThreadTimer.stop()
            self.feedbackThreadTimer.stop()
            self.eeglogger.stopBoard()
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
            self.indicatorAPP.actualizar_orden("Fijar la mirada en la cruz...")
            #Generamos un número aleatorio entre self.startingTimes[0] y self.startingTimes[1], redondeado a 1 decimal
            startingTime = round(random.uniform(self.startingTimes[0], self.startingTimes[1]), 1)
            self.__startingTime = startingTime
            startingTime = int(startingTime * 1000) #lo pasamos a milisegundos
            self.__trialPhase = 1 # pasamos a la siguiente fase -> CUE
            self.eegThreadTimer.setInterval(startingTime) #esperamos el tiempo aleatorio

        elif self.__trialPhase == 1:
            self.indicatorAPP.showCruz(False) #desactivamos la cruz
            logging.info("Iniciamos fase cue del trial")
            claseActual = self.trialsSesion[self.__trialNumber]
            classNameActual = self.clasesNames[self.classes.index(claseActual)]
            self.indicatorAPP.actualizar_orden(f"{classNameActual}", fontsize = 46,
                                              background = "rgb(38,38,38)", font_color = "white")
            self.__trialPhase = 2 # la siguiente fase es la de finalización del trial
            self.eegThreadTimer.setInterval(int(self.cueDuration * 1000))

        elif self.__trialPhase == 2:
            logging.info("Iniciamos fase de finalización del trial")
            self.indicatorAPP.actualizar_orden("Fin de tarea...")
            self.__trialPhase = -1 #Fase para guardar datos de EEG
            self.eegThreadTimer.setInterval(int(self.finishDuration * 1000))

        else:
            #Al finalizar el trial, guardamos los datos de EEG
            logging.info("Guardando datos de EEG")
            newData = self.eeglogger.getData(self.__startingTime + self.cueDuration + self.finishDuration, removeDataFromBuffer=False)[self.channels]
            self.eeglogger.saveData(newData, fileName = self.eegFileName, path = self.eegStoredFolder, append=True)
            self.saveEvents() #guardamos los eventos de la sesión
            self.__trialPhase = 0 #volvemos a la fase inicial del trial
            self.__trialNumber += 1 #incrementamos el número de trial
            self.eegThreadTimer.setInterval(1)

    def feedbackThread(self):
        """Función para hilo de lectura de EEG durante fase de entrenamiento.
        Sólo se almacena trozos de EEG correspondientes a la suma de startTrainingTime y cueDuration.
        """

        if self.__trialPhase == 0:
            print(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            logging.info(f"Trial {self.__trialNumber + 1} de {len(self.trialsSesion)}")
            self.indicatorAPP.showCruz(True) #mostramos la cruz
            self.indicatorAPP.actualizar_orden("Fijar la mirada en la cruz...")
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
            self.indicatorAPP.actualizar_orden(f"{classNameActual}", fontsize = 46,
                                              background = "rgb(38,38,38)", font_color = "white")
            self.__trialPhase = 2 # la siguiente fase es la de finalización del trial
            self._dataToClasify = self.eeglogger.getData(self.cueDuration, removeDataFromBuffer=False)[self.channels]
            self.classifyEEGTimer.start()
            # self.feedbackThreadTimer.setInterval(int(self.cueDuration * 1000))
            self.feedbackThreadTimer.setInterval(int((self.cueDuration + self.lenToClassify*0.05) * 1000))

        elif self.__trialPhase == 2:
            self.classifyEEGTimer.stop() #detenemos el timer de clasificación
            logging.info("Iniciamos fase de finalización del trial")
            self.indicatorAPP.actualizar_orden("Fin de tarea...")
            self.__trialPhase = -1 #volvemos a la fase inicial del trial
            self.feedbackThreadTimer.setInterval(int(self.finishDuration * 1000))

        else:
            #Al finalizar el trial, guardamos los datos de EEG
            logging.info("Guardando datos de EEG")
            newData = self.eeglogger.getData(self.__startingTime + self.cueDuration + self.finishDuration, removeDataFromBuffer=False)[self.channels]
            self.eeglogger.saveData(newData, fileName = self.eegFileName, path = self.eegStoredFolder, append=True)
            self.saveEvents() #guardamos los eventos de la sesión
            self.__trialPhase = 0 #volvemos a la fase inicial del trial
            self.__trialNumber += 1 #incrementamos el número de trial
            self.eegThreadTimer.setInterval(1)

    def showGUIAPPs(self):
        """Función para configurar la sesión de entrenamiento usando self.confiAPP.
        """
        self.indicatorAPP.show() #mostramos la APP
        self.supervisionAPP.show() #mostramos la APP
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
        ##TODO: actualizar la APP de supervisión
        pass

    def classifyEEG(self):
        """Función para clasificar EEG
        La función se llama cada vez que se activa el timer self.classifyEEGTimer. La duración
        del timer esta dada por self.classifyEEGTimer.setInterval(self.lenToClassify*1000).

        Se obtiene un nuevo trozo de EEG de longitud self.lenToClassify segundos, se añade al
        buffer de datos a clasificar y se clasifica. El resultado de la clasificación se almacena
        en self.prediction.

        Por cada entrada a la función, se elimina el primer trozo de datos del buffer de datos a
        clasificar y se añade el nuevo trozo de datos. La idea es actualizar los datos mientras la persona ejecuta
        la tarea.
        """
        newData = self.eeglogger.getData(self.lenToClassify, removeDataFromBuffer = False)[self.channels]
        samplesToRemove = int(self.lenToClassify*self.sample_rate)
        self._dataToClasify = np.concatenate((self._dataToClasify[:,samplesToRemove:], newData), axis = 1)
        channels, samples = self._dataToClasify.shape
        #camibamos la forma de los datos para que sea compatible con el modelo
        trialToPredict = self._dataToClasify.reshape(1,channels,samples)
        self.prediction = self.pipeline.predict(trialToPredict) #aplicamos data al pipeline
        self.probas = self.pipeline.predict_proba(trialToPredict) #obtenemos las probabilidades de cada clase
        logging.info("Dato clasificado", self.prediction)
        
    def start(self):
        """Método para iniciar la sesión"""
        print(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")
        logging.info(f"Preparando sesión {self.sesionNumber} del sujeto {self.subjectName}")
        if self.typeSesion == 0:
            # self.indicatorAPP.show() #mostramos la APP
            self.indicatorAPP.actualizar_orden("Iniciando sesión de entrenamiento") #actualizamos app
        
        if self.typeSesion == 1:
            # self.indicatorAPP.show() #mostramos la APP
            self.indicatorAPP.actualizar_orden("Iniciando sesión de feedback") #actualizamos app
        self.iniSesionTimer.start()

    def startSesion(self):
        """Método para iniciar timers del Core"""
        self.iniSesionTimer.stop()
        self.setFolders(rootFolder = self.rootFolder) #configuramos las carpetas de almacenamiento
        self.saveConfigParameters(self.eegStoredFolder+self.eegFileName[:-4]+"_config.json") #guardamos los parámetros de configuración
        
        if self.typeSesion == 0:
            print("Inicio de sesión de entrenamiento")
            logging.info("Inicio de sesión de entrenamiento")
            self.setEEGLogger()
            self.makeAndMixTrials()
            self.checkTrialsTimer.start()
            self.eegThreadTimer.start() #iniciamos timer para controlar hilo entrenamiento
            
        elif self.typeSesion == 1:
            self.setEEGLogger()
            if not self.__customPipeline:
                self.setPipeline() #cargamos pipeline desde archivo. ESTO ES LO RECOMENDABLE
            else: #si se ha seleccionado un pipeline personalizado, lo cargamos
                self.setFilter() #seteamos filtro
                # self.setCSP() #seteamos CSP
                # self.RavelTransformer() #seteamos RavelTransformer
                # self.setClassifier() #seteamos clasificador
                self.setPipeline(filter = self.filter) #seteamos pipeline
            print("Inicio de sesión de Feedback")
            self.makeAndMixTrials()
            self.checkTrialsTimer.start()
            self.feedbackThreadTimer.start() #iniciamos timer para controlar hilo entrenamiento

        elif self.typeSesion == 2:
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

    #Creamos un diccionario con los parámetros de configuración iniciales
    parameters = {
        "typeSesion": 1, #0: Entrenamiento, 1: Feedback, 2: Online
        "cueType": 0, #0: Se ejecutan movimientos, 1: Se imaginan los movimientos
        "classes": [1, 2, 3, 4, 5], #Clases a clasificar
        "clasesNames": ["mover Mano Izquierda", "mover Mano Derecha", "mover Ambas Manos", "mover Ambos Pies", "Rest"], #MI: Mano izquierda, MD: Mano derecha, AM: Ambas manos, AP: Ambos pies, R: Reposo
        "ntrials": 1, #Número de trials por clase
        "startingTimes": [2, 2.5], #Tiempos para iniciar un trial de manera aleatoria entre los extremos, en segundos
        "cueDuration": 4, #En segundos
        "finishDuration": 3, #En segundos
        "lenToClassify": 1.0, #Trozo de señal a clasificar, en segundos
        "subjectName": "subject_test", #nombre del sujeto
        "sesionNumber": 1, #número de sesión
        "boardParams": { 
            "boardName": "synthetic", #Board de registro
            "channels": [13,14,15], #[0, 1, 2, 3, 4, 5, 6, 7], #Canales de registro
            "serialPort": "COM5" #puerto serial
        },
        "filterParameters": {
            "lowcut": 8., #Frecuencia de corte baja
            "highcut": 28., #Frecuencia de corte alta
            "notch_freq": 50., #Frecuencia corte del notch
            "notch_width": 1, #Ancho de del notch
            "sample_rate": 250., #Frecuencia de muestreo
            "axisToCompute": 2,
        },
        "featureExtractorMethod": "welch",
        "rootFolder": "data",
        "cspFile": "data/dummyTest/csps/dummycsp.pickle",
        "classifierFile": "data/dummyTest/classifiers/dummyclassifier.pickle",
        "customPipeline": False,
        "pipelineFile": "data/dummyTest/pipelines/best_estimator_svm.pkl",
    }

    app = QApplication(sys.argv)

    core = Core(parameters, ConfigAPP("config.json"), IndicatorAPP(), SupervisionAPP())

    sys.exit(app.exec_())

    # import numpy as np

    # data = np.load("data/subjetc_test/eegdata/sesion1/sn1_ts1_ct1_r1.npy")

    # data.shape