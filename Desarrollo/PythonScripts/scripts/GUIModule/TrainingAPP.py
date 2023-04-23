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
        self.Gestionar_sesion(True) #Si seleccionas False no hará ruido al instanciar la interfaz
        
    def actualizar_orden(self, texto):
        """
        Actualiza la etiqueta que da la orden
            texto (str): texto de la orden
        """
        self.label_orden.setText(texto)

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
        
    def Gestionar_sesion(self, mostrar:bool):
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