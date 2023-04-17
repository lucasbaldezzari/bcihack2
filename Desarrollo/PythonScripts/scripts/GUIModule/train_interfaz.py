from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import pyqtgraph as pg
import sys

class entrenamiento(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi("registro.ui", self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = entrenamiento()
    _ventana.show()
    app.exec_()