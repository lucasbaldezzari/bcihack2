from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import pyqtgraph as pg
import sys
import os

class entrenamiento(QDialog):

    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), 'registro.ui')
        uic.loadUi(ui_path, self)
        
    def actualizar_orden(self, texto):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        self.label_orden.setText(texto)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = entrenamiento()
    _ventana.show()
    app.exec_()