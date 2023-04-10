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

    def __init__(self, gui, gui2):
        QThread.__init__(self)
        self.gui = gui
        self.gui2 = gui2

    def run(self):
        while True:
            # Realizar tareas para actualizar la interfaz aquí
            print("Actualizando interfaz")
            
            # if self.gui.isVisible():
                

            # Emitir la señal de actualización
            self.senal_actualizacion.emit()

            # Esperar un tiempo para evitar consumir demasiado CPU
            self.sleep(1)

app = QApplication(sys.argv) 
gui = MainWindow()
gui2 = entrenamiento()
gui.show()

# def abrir_nueva_ventana():
#     nueva_ventana = entrenamiento()
#     nueva_ventana.show()

# gui.closeEvent = abrir_nueva_ventana()

# Instanciar el hilo de actualización de interfaz
hilo_actualizacion = ActualizacionInterfaz(gui, gui2)
# Conectar la señal de actualización al método de la interfaz que realiza la actualización
hilo_actualizacion.senal_actualizacion.connect(gui.actualizar_interfaz)
hilo_actualizacion.start()

sys.exit(app.exec_())