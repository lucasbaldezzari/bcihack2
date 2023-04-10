from registro_config import MainWindow
from train_interfaz import entrenamiento
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from PyQt5 import uic
import time
import pyqtgraph as pg
import sys

import argparse
import logging

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import fileAdmin as fa
from DataThread import DataThread as DT

import queue
import threading
import time

class ActualizacionInterfaz(QThread):
    # Señal para indicar que se ha actualizado la interfaz
    senal_actualizacion = pyqtSignal()

    def __init__(self, gui):
        QThread.__init__(self)
        self.gui = gui

    def run(self):
        while True:
            print('Actualizando Interfaz')
            
            self.sleep(1)
        
    #     self.ThreadActive = True

    #     contador_trials = 0
    #     trials = self.trials * self.average_trials #TRIALS TOTALES por accion
        
    #     EEGdata = []
    #     EEGTrialsAveraged = []

    #     path = "recordedEEG" #directorio donde se almacenan los registros de EEG.

    #     """Datos del sujeto, la sesión y la corrida"""
    #     generalInformation = f'{self.placaBCI}. Duración trial {self.trial_duration}'

    #     BoardShim.enable_dev_board_logger()
    #     logging.basicConfig(level=logging.DEBUG)
        
    #     placas = {"cyton": BoardIds.CYTON_BOARD.value, #IMPORTANTE: frecuencia muestreo 256Hz
    #             "ganglion": BoardIds.GANGLION_BOARD.value, #IMPORTANTE: frecuencia muestro 200Hz
    #             "synthetic": BoardIds.SYNTHETIC_BOARD.value}
        
    #     placa = placas[f"{self.placaBCI}"]  
    #     electrodos = "pasivos"
        
    #     puerto = self.puertoBCI #Chequear el puerto al cual se conectará la placa
        
    #     parser = argparse.ArgumentParser()
        
    #     # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    #     parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
    #                         default=0)
    #     parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    #     parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
    #                         default=0)
    #     parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')

    #     #IMPORTENTE: Chequear en que puerto esta conectada la OpenBCI. En este ejemplo esta en el COM4    
    #     # parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM4')
    #     parser.add_argument('--serial-port', type=str, help='serial port', required=False, default = puerto)
    #     parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    #     parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    #     parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    #     parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    #     parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #                         required=False, default = placa)
    #     # parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #     #                     required=False, default=BoardIds.CYTON_BOARD)
    #     parser.add_argument('--file', type=str, help='file', required=False, default='')
    #     args = parser.parse_args()

    #     params = BrainFlowInputParams()
    #     params.ip_port = args.ip_port
    #     params.serial_port = args.serial_port
    #     params.mac_address = args.mac_address
    #     params.other_info = args.other_info
    #     params.serial_number = args.serial_number
    #     params.ip_address = args.ip_address
    #     params.ip_protocol = args.ip_protocol
    #     params.timeout = args.timeout
    #     params.file = args.file

    #     board_shim = BoardShim(args.board_id, params) #genero un objeto para control de placas de Brainflow

        

    #     board_shim.prepare_session()
    #     time.sleep(2) #esperamos 2 segundos

    #     #### CONFIGURAMOS LA PLACA CYTON O GANGLION######
    #     """
    #     IMPORTANTE: No tocar estos parámetros.
    #     El string es:
    #     x (CHANNEL, POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET) X

    #     Doc: https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
    #     Doc: https://docs.openbci.com/Ganglion/GanglionSDK/
    #     """

    #     if placa == BoardIds.GANGLION_BOARD.value:
    #         canalesAdesactivar = ["3","4"]
    #         for canal in canalesAdesactivar:
    #             board_shim.config_board(canal) #apagamos los canales 3 y 4
    #             time.sleep(1)

    #     if placa == BoardIds.CYTON_BOARD.value:
    #         if electrodos == "pasivos":
    #             configCanalesCyton = {
    #                 "canal1": "x1160110X", #ON|Ganancia 24x|Normal input|Connect from Bias|
    #                 "canal2": "x2060110X", #ON|Ganancia 24x|Normal input|Connect from Bias|
    #                 "canal3": "x3101000X", #Canal OFF
    #                 "canal4": "x4101000X", #Canal OFF
    #                 "canal5": "x5101000X", #Canal OFF
    #                 "canal6": "x6101000X", #Canal OFF
    #                 "canal7": "x7101000X", #Canal OFF
    #                 "canal8": "x8101000X", #Canal OFF
    #             }
    #             for config in configCanalesCyton:
    #                 board_shim.config_board(configCanalesCyton[config])
    #                 time.sleep(0.5)

    #         if electrodos == "activos":
    #             configCanalesCyton = {
    #                 "canal1": "x1040110X", #ON|Ganancia 8x|Normal input|Connect from Bias|
    #                 "canal2": "x2040110X", #ON|Ganancia 8x|Normal input|Connect from Bias|
    #                 "canal3": "x3101000X", #Canal OFF
    #                 "canal4": "x4101000X", #Canal OFF
    #                 "canal5": "x5101000X", #Canal OFF
    #                 "canal6": "x6101000X", #Canal OFF
    #                 "canal7": "x7101000X", #Canal OFF
    #                 "canal8": "x8101000X", #Canal OFF
    #             }
    #             for config in configCanalesCyton:
    #                 board_shim.config_board(configCanalesCyton[config])
    #                 time.sleep(0.5)

    #     board_shim.start_stream(450000, args.streamer_params) #iniciamos OpenBCI. Ahora estamos recibiendo datos.
    #     time.sleep(2) #esperamos 4 segundos
        
    #     data_thread = DT(board_shim, args.board_id) #genero un objeto DataThread para extraer datos de la OpenBCI
    #     time.sleep(2)

    #     if self.placaBCI == 'synthetic':
    #         fm = 250

    #     else:
    #         fm = board_shim.get_sampling_rate()

    #     # # print(fm)
    #     # channels = 2
    #     #channels = len(BoardShim.get_eeg_channels(args.board_id))
    #     #samplePoints = int(fm*self.trial_duration)

    #     datosSession = {
    #                 'subject': self.sujeto,
    #                 'date': self.fecha,
    #                 'generalInformation': generalInformation,
    #                 'trialDuration': self.trial_duration,
    #                 'tiempo_accion': self.tiempo_accion,
    #                 'tiempo_descanso': self.tiempo_descanso,
    #                 'tiempo_preparacion': self.tiempo_preparacion,
    #                 'channelsRecorded': self.canales, 
    #                 #'dataShape': [channelsRecorded, samplePoints, trials],
    #                 'eeg': None
    #                     }

    #     acciones = {
    #         1: 'Imagina que mueves la mano derecha',
    #         2: 'Imagina que mueves la mano izquierda',
    #         3: 'Imagina que mueves el pie derecho',
    #         4: 'Imagina que mueves el pie izquierdo',
    #     }

    #     eegs = {}

    #     time.sleep(1) 

    #     start = time.time()
    #     start2 = time.time()

    #     estado_actual = 0

    #     contador_trial_promedios = 0
    #     contador_trial_sinpromedios = 0
    #     tiempo_anterior = 0

    #     self.update.emit(f'Trial número {contador_trials+1}')

    #     try:
    #         for accion in range(len(acciones)):

    #             self.update3.emit(acciones[accion+1])

    #             while trials > contador_trials:
    #                 actual = time.time() - start

    #                 if self.tiempo_preparacion >= actual and actual >= 0 and estado_actual != 0:
    #                     estado_actual = 0
    #                     self.update.emit(f'Trial número {contador_trial_sinpromedios+1}, Promedio {contador_trial_promedios+1}, PREPARACION')

    #                 if actual >= self.tiempo_preparacion and (self.tiempo_preparacion + self.tiempo_accion) >= actual and estado_actual != 1:
    #                     estado_actual = 1
    #                     self.update.emit(f'Trial número {contador_trial_sinpromedios+1}, Promedio {contador_trial_promedios+1}, ACCION')

    #                 if actual >= (self.tiempo_preparacion + self.tiempo_accion) and self.trial_duration >= actual and estado_actual != 2:
    #                     estado_actual = 2
    #                     self.update.emit(f'Trial número {contador_trial_sinpromedios+1}, Promedio {contador_trial_promedios+1}, DESCANSO')

    #                 if actual >= self.trial_duration:
    #                     contador_trial_promedios += 1
    #                     contador_trials += 1
    #                     currentData = data_thread.getData(self.trial_duration, channels = self.canales)
    #                     # print(currentData.shape)
    #                     EEGTrialsAveraged.append(currentData)
    #                     #self.update4.emit(currentData)
    #                     # print(np.asarray(EEGTrialsAveraged).shape)
    #                     start = time.time()

    #                     if contador_trials % self.average_trials == 0 and contador_trials != 0:
    #                         contador_trial_sinpromedios += 1
    #                         print(f'trial {contador_trial_sinpromedios}')
    #                         EEGdata.append(np.asarray(EEGTrialsAveraged).mean(axis = 0))
    #                         # print(np.array(EEGdata).shape)
    #                         EEGTrialsAveraged = []
    #                         contador_trial_promedios = 0

    #                 if (time.time() - start2) >= 0.01:
    #                     start2 = self.actualizar_barra(actual, estado_actual, self.trial_duration) 
    #                     tiempo = np.arange(tiempo_anterior, tiempo_anterior+0.01, 1/fm)  
    #                     self.update4.emit('eeg', tiempo, data_thread.getData(0.01, channels = self.canales))
    #                     tiempo_anterior = tiempo[-1]


    #                 if self.control_bucle:
    #                     break
                
    #             if self.control_bucle:
    #                     break

    #             contador_trials = 0 
    #             contador_trial_sinpromedios = 0
    #             # print(f'La accion es {accion+1}')
    #             # print(np.array(EEGdata).shape)
    #             try:
    #                 eegs[accion+1] = np.reshape(np.array(EEGdata), (self.trials, len(self.canales), 
    #                 int(fm*self.trial_duration)))[ : , : ,int(fm*self.tiempo_preparacion):int(fm*(self.tiempo_preparacion+self.tiempo_accion))]
    #             except:
    #                 eegs[accion+1] = np.reshape(np.array(EEGdata), (self.trials, self.canales, 
    #                 int(fm*self.trial_duration)))[ : , : ,int(fm*self.tiempo_preparacion):int(fm*(self.tiempo_preparacion+self.tiempo_accion))]
    #             print(eegs[accion+1].shape)
    #             EEGdata = []

    #     except BaseException as e:
    #         logging.warning('Exception', exc_info=True)
            
    #     finally:
    #         if board_shim.is_prepared():
    #             logging.info('Releasing session')
    #             board_shim.release_session()

    #         datosSession["eeg"] = eegs
    #         # print(datosSession.keys())
    #         # print(datosSession["eeg"].keys())
    #         fa.saveData(path = path, dictionary = datosSession, fileName = datosSession["subject"])

    #         self.update.emit('Finalizada')
    #         start2 = self.actualizar_barra(self.trial_duration, 2, self.trial_duration) 

    # def actualizar_barra(self, tiempo_actual, estado, tiempo_total):
    #     self.update2.emit(tiempo_actual, estado, tiempo_total)
    #     start2 = time.time()

    #     return start2

app = QApplication(sys.argv)

gui2 = entrenamiento()
hilo_actualizacion_gui2 = ActualizacionInterfaz(gui2)

gui = MainWindow(gui2, hilo_actualizacion_gui2)

gui.show()
sys.exit(app.exec_())