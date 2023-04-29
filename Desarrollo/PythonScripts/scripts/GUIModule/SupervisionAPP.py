from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import pyqtgraph as pg
import sys
import os
import numpy as np

class SupervisionAPP(QDialog):
    """
    Interfaz gráfica para supervisar las sesiones
    """
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), 'supervision.ui')
        uic.loadUi(ui_path, self)
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.x = np.array([0])
        self.y = np.array([0])

    def actualizar_orden(self, texto):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        self.label_orden.setText(texto)

    def actualizar_barra(self, tiempo_total, tiempo_actual, etapa):
        """
        Actualiza la barra de progreso del trial
            tiempo_actual (float): tiempo actual del trial en segundos. No debe ser mayor al tiempo de trial total
            etapa (int): etapa actual cuando se actualiza la barra. 
                0: preparacion; 
                1: accion; 
                2: descanso;
        """
        try:
            self.progressBar.setValue(int(tiempo_actual*100/tiempo_total))

            if etapa == 0:
                self.progressBar.setStyleSheet("QProgressBar::chunk {background-color: green;}")

            if etapa == 1:
                self.progressBar.setStyleSheet("QProgressBar::chunk {background-color: orange;}")

            if etapa == 2:
                self.progressBar.setStyleSheet("QProgressBar::chunk {background-color: red;}")
        
        except:
            pass

    def actualizar_grafica(self, name, tiempo, datos):
        """
        Actualiza el grafico de Canal Tension vs Tiempo
            name (str): nombre del canal
            tiempo (list): vector tiempo adquirido
            datos (list): vector de tension adquirido
        """
        if len(self.x) > 1000:
            self.x = np.concatenate((tiempo[:int(len(datos[0]))], self.x[:-int(len(datos[0]))]))
            self.y = np.concatenate((datos[0], self.y[:-int(len(datos[0]))])) 
        else:
            self.x = np.concatenate((self.x, tiempo[:int(len(datos[0]))]))
            self.y = np.concatenate((self.y, datos[0])) 
        
        if name in self.traces:
            self.traces[name].setData(self.x, self.y)
        else:
            self.traces[name] = self.graphicsView.plot(pen='w')
    
    def actualizar_info(self, sesion:int, trial:float, etapa:int, canales:str):
        """
        Para actualizar los campos de información en la interfaz de superivisión
        """
        self.label_sesion.setText(f'Tipo de Sesión: {sesion}')
        self.label_trial.setText(f'Tiempo del Trial: {trial} s')
        self.label_etapa.setText(f'Estapa Actual: {etapa}')
        self.label_etapa.setText(f'Canales Seleccionados: {canales}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = SupervisionAPP()
    _ventana.show()
    app.exec_()