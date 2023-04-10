from registro_configv3 import MainWindow
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

    def __init__(self, gui, gui2):
        QThread.__init__(self)
        self.gui = gui
        self.gui2 = gui2
        self.flag_config_on = False #Comprueba si se cerro la ventana de configuracion
        self.gui.show()

    def run(self):
        

        while True:
            print('Actualizando Interfaz')

            if self.gui.isVisible():
                self.flag_config_on = True
                print('hola')

            elif self.gui.isVisible() == False and self.flag_config_on:
                self.gui2.show() 
                print('hola2')
                self.flag_config_on = False

            elif self.gui2.isVisible():
                pass
                
            self.sleep(1)

app = QApplication(sys.argv)

gui = MainWindow()

gui2 = entrenamiento()

hilo = ActualizacionInterfaz(gui, gui2)

hilo.start()

sys.exit(app.exec_())