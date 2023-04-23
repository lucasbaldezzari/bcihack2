from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import sys
import os
import winsound

class TrainingAPP(QDialog):
    """
    Interfaz gráfica que tiene como fin únicamente mostrar la orden para la adquisición de datos 
    de entrenamiento para el clasificador
    """
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), 'registro.ui')
        uic.loadUi(ui_path, self)
        self.Centrar(self.cruz)
        self.showCruz(False) #Si seleccionas False no hará ruido al instanciar la interfaz

        #obtenemos el background de label_orden
        self.background_color = self.label_orden.palette().color(QPalette.Background).name()
        self.font_color = "rgb(25,50,200)" #self.label_orden.palette().color(QPalette.Base)

    def actualizar_orden(self, texto, fontsize = 36, background = None, border = "1px", font_color = "black"):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        self.label_orden.setFont(QFont('Berlin Sans', fontsize))
        self.label_orden.setText(texto)
        if background:
            self.label_orden.setStyleSheet(f"background-color: {background};border: {border} solid black;color: {font_color}")
        else:
            self.label_orden.setStyleSheet(f"background-color: {self.background_color}; border: 0px solid black;color: {self.font_color}")

    def Centrar(self, objeto):
        """
        Centra el objeto (widget) en la pantalla
        """
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()

        # Establecer las dimensiones de la ventana
        self.setGeometry(0, 0, width, height)

        # Centrar el QLabel en la ventana
        objeto.setGeometry(int(width/2 - objeto.width()/2), int(height/2 - objeto.height()/2), 
                           int(objeto.width()),int(objeto.height()))
        
    def Subir(self, objeto):
        """
        Sube el objeto sobre el centro de la pantalla
        """
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()

        # Establecer las dimensiones de la ventana
        self.setGeometry(0, 0, width, height)

        # Centrar el QLabel en la ventana
        objeto.setGeometry(int(width/2 - objeto.width()/2), int(height/2 - objeto.height()/2) - 100, 
                           int(objeto.width()),int(objeto.height()))
        
    def showCruz(self, mostrar:bool):
        """
        Muestra o no la cruz de preparación en la interfaz
        """
        if mostrar:
            self.cruz.setVisible(True)
            self.Subir(self.label_orden)
            winsound.Beep(440, 1000)
        else:
            self.cruz.setVisible(False)
            self.Centrar(self.label_orden)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = TrainingAPP()
    _ventana.showMaximized()
    app.exec_()