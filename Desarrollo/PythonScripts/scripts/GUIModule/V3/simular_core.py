from registro_config import MainWindow
from train_interfaz import entrenamiento
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

directorio =  'C:/Users/Emi/Documents/Repos/bcihack2/Desarrollo/PythonScripts/scripts/config.json' #establesca donde este config.json

app = QApplication(sys.argv)

gui2 = entrenamiento()

gui = MainWindow(gui2, directorio)

def update():
    if gui2.isVisible():
        gui2.label_orden.setText(f'Holaa')

phaseTrialTimer = QTimer() #Timer para control de tiempo de las fases de trials
phaseTrialTimer.setInterval(1000) #1 milisegundo sólo para el inicio de sesión.
phaseTrialTimer.timeout.connect(update)
phaseTrialTimer.start()

gui.show()
sys.exit(app.exec_())