from registro_config import MainWindow

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

class entrenamiento(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi("registro.ui", self)
        
        # self.btn_control.clicked.connect(self.Control)
        # self.btn_regresar.clicked.connect(self.Cerrar)
        # pg.setConfigOptions(antialias=True)
        # # Create a PlotWidget
        # plot = pg.PlotWidget()

        # # # Plot data
        # # plot.plot(data)

        # # Create a GraphicsScene
        # scene = pg.GraphicsScene()

        # # Add the PlotWidget to the GraphicsScene
        # scene.addWidget(plot)

        # # Set the GraphicsScene to the GraphicsView
        # self.graphicsView.setScene(scene)

        # self.curve = pg.PlotCurveItem(pen='r')
        # plot.addItem(self.curve)
        # self.traces = dict()
        # self.x = np.array([0])
        # self.y = np.array([0])

    def Control(self):
        if self.btn_control.text() == 'Iniciar':
            self.btn_control.setText('Detener')
            #self.btn_control.setStyleSheet("background-color : red")
            self.btn_regresar.setEnabled(False)
            sujeto = self.line_sujeto.text()
            fecha = self.line_fecha.text()
            puerto = self.line_puerto.text()
            tiempo_preparacion = float(self.line_tiempo_preparacion.text())
            tiempo_accion = float(self.line_tiempo_accion.text())
            tiempo_descanso = float(self.line_tiempo_descanso.text())
            trials = int(self.line_trials.text())
            trials_promedio = int(self.line_promedio.text())
            try:
                canales = int(self.line_canales.text())
            except:
                canales = eval(self.line_canales.text())

            placa = self.desplegable_placa.currentText()
            self.data = np.array([0])
            # self.Worker1 = Worker1(sujeto, fecha, placa, puerto, tiempo_preparacion, tiempo_accion, tiempo_descanso,
            #                         trials, trials_promedio, canales)
            self.Worker1.start()
            self.Worker1.update.connect(self.Actualizar)
            self.Worker1.update2.connect(self.Barra)
            self.Worker1.update3.connect(self.Accion)
            self.Worker1.update4.connect(self.Graficar)
            
            
        else:
            self.btn_control.setText('Iniciar')
            #self.btn_control.setStyleSheet("background-color : green")
            self.btn_regresar.setEnabled(True)
            self.Worker1.control_bucle = True
        

    def Cerrar(self):
        self.close()
        #Ventana.exec_()

    # def Actualizar(self, informacion):
    #     self.label_orden.setText(informacion)

    # def Barra(self, valor, etapa, tiempo_trial):
    #     try:
    #         self.progressBar.setValue(int(valor*100/tiempo_trial))

    #         if etapa == 0:
    #             self.progressBar.setStyleSheet("QProgressBar::chunk {background-color: green;}")

    #         if etapa == 1:
    #             self.progressBar.setStyleSheet("QProgressBar::chunk {background-color: orange;}")

    #         if etapa == 2:
    #             self.progressBar.setStyleSheet("QProgressBar::chunk {background-color: red;}")
     
    #     except:
    #         pass

    def Accion(self, accion):
        self.label_accion.setText(accion)

    # def Graficar(self, name, tiempo, datos):
    #     # self.data = np.roll(self.data, -1)
    #     # self.data[-1] = datos[0]
    #     # self.curve.setData(tiempo[:int(len(datos[0]))], self.data[:int(len(datos[0]))])
    #     # print(datos.shape)
    #     # self.x = np.concatenate((self.x, tiempo[:int(len(datos[0]))]))
    #     # self.y = np.concatenate((self.y, datos[0])) 

    #     if len(self.x) > 1000:
    #         self.x = np.concatenate((tiempo[:int(len(datos[0]))], self.x[:-int(len(datos[0]))]))
    #         self.y = np.concatenate((datos[0], self.y[:-int(len(datos[0]))])) 
    #     else:
    #         self.x = np.concatenate((self.x, tiempo[:int(len(datos[0]))]))
    #         self.y = np.concatenate((self.y, datos[0])) 
         
    #     if name in self.traces:
    #         self.traces[name].setData(self.x, self.y)
    #     else:
    #         self.traces[name] = self.graphicsView.plot(pen='y')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = entrenamiento()
    _ventana.show()
    app.exec_()