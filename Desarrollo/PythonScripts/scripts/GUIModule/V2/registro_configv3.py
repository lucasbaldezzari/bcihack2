from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from PyQt5 import uic
import sys
import pickle

class MainWindow(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi("config_registro.ui", self)
        
        self.btn_control.clicked.connect(self.Control)
        # self.btn_regresar.clicked.connect(self.Cerrar)
        # pg.setConfigOptions(antialias=True)

    def Control(self):

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

        self.guardar_diccionario(sujeto, fecha, placa, puerto, tiempo_preparacion, tiempo_accion, tiempo_descanso,
                                trials, trials_promedio, canales)

        self.close()


    def guardar_diccionario(self, sujeto, fecha, placa, puerto, tiempo_preparacion, tiempo_accion, tiempo_descanso,
                                trials, trials_promedio, canales):
        diccionario = dict()
        diccionario['sujeto'] = sujeto
        diccionario['fecha'] = fecha
        diccionario['placa'] = placa
        diccionario['puerto'] = puerto
        diccionario['tiempo_preparacion'] = tiempo_preparacion
        diccionario['tiempo_accion'] = tiempo_accion
        diccionario['tiempo_descanso'] = tiempo_descanso
        diccionario['trials'] = trials
        diccionario['trials_promedio'] = trials_promedio
        diccionario['canales'] = canales

        # Abrimos un archivo en modo de escritura binario
        with open(f'parametros_sesiones/configuracion_{sujeto}', 'wb') as archivo:

            # Usamos la función dump para guardar el diccionario en el archivo
            pickle.dump(diccionario, archivo)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = MainWindow()
    _ventana.show()
    app.exec_()