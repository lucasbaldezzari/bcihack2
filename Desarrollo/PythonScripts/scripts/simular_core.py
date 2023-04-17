from GUIModule.registro_config import MainWindow
from GUIModule.train_interfaz import entrenamiento
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

directorio =  'config.json' #establesca la ruta donde se encuentre su archivo config.json

app = QApplication(sys.argv)

gui2 = entrenamiento()

gui = MainWindow(gui2, directorio) #gui una vez que se cierra muestra gui2

def update():
    if gui2.isVisible(): #Si la gui2 se esta mostrando en pantalla
        gui2.label_orden.setText(f'Holaa') 

phaseTrialTimer = QTimer() #Timer para control de tiempo de las fases de trials
phaseTrialTimer.setInterval(1000) #1 milisegundo sólo para el inicio de sesión.
phaseTrialTimer.timeout.connect(update)
phaseTrialTimer.start()

gui.show()
sys.exit(app.exec_())