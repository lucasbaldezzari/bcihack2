from GUIModule.registro_config import MainWindow
from GUIModule.train_interfaz import entrenamiento
from GUIModule.interfaz_supervision import supervision
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

directorio =  'config.json' #establesca la ruta donde se encuentre su archivo config.json

tiempo_trial = 10

app = QApplication(sys.argv)

gui_entrenamiento = entrenamiento()

gui_supervision = supervision(tiempo_trial)

gui = MainWindow(gui_entrenamiento, gui_supervision, directorio) #gui una vez que se cierra muestra gui2

def update_entrenamiento(orden):
    if gui_entrenamiento.isVisible(): #Si la gui2 se esta mostrando en pantalla
        print("Actualizando entrenamiento")
        gui_entrenamiento.actualizar_orden(orden) 

def update_supervision(orden, tiempo_actual, etapa):
    if gui_supervision.isVisible(): #Si la gui2 se esta mostrando en pantalla
        print("Actualizando supervision")
        gui_supervision.actualizar_orden(orden) 
        gui_supervision.actualizar_barra(tiempo_actual, etapa)

actualizar_entrenamiento = QTimer() #Timer para control 
actualizar_entrenamiento.setInterval(1000) 
actualizar_entrenamiento.timeout.connect(lambda orden = "Mover mano derecha": update_entrenamiento(orden))
actualizar_entrenamiento.start()

actualizar_supervision = QTimer() #Timer para control 
actualizar_supervision.setInterval(1000) 
actualizar_supervision.timeout.connect(lambda orden = "Mover mano derecha", tiempo_actual = 3,
                                        etapa = 1: update_supervision(orden, tiempo_actual, etapa))
actualizar_supervision.start()

gui.show()
sys.exit(app.exec_())