from GUIModule.ConfigAPP import ConfigAPP
from GUIModule.IndicatorAPP import IndicatorAPP
from GUIModule.SupervisionAPP import SupervisionAPP
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

directorio =  'config.json' #establesca la ruta donde se encuentre su archivo config.json

app = QApplication(sys.argv)

gui_entrenamiento = IndicatorAPP()

gui_supervision = SupervisionAPP()

gui = ConfigAPP(directorio)

def update_entrenamiento(orden:str): #Para actualizar la interfaz que muestra la orden al sujeto
    # print("Actualizando entrenamiento")
    gui_entrenamiento.actualizar_orden(orden) 

def update_supervision(orden:str, tiempo_total:float, tiempo_actual:float, etapa:int): #Actualizar la interfaz de supervisión
    # print("Actualizando supervision")
    gui_supervision.actualizar_orden(orden) 
    gui_supervision.actualizar_barra(tiempo_total, tiempo_actual, etapa)

actualizar_entrenamiento = QTimer() #Timer actualizar la app de entrenamiento. Se inicia una vez se muestra la interfaz
actualizar_entrenamiento.setInterval(1000) 
actualizar_entrenamiento.timeout.connect(lambda orden = "Mover mano derecha": update_entrenamiento(orden))

actualizar_supervision = QTimer() #Timer para actualizar la app de supervision. Se inicia una vez se muestra la interfaz
actualizar_supervision.setInterval(1000) 
actualizar_supervision.timeout.connect(lambda orden = "Mover mano derecha", tt = 10, ta = 5, 
                                       et = 2: update_supervision(orden, tt, ta, et))

def iniciar_guis():
    """
    Supervisa que se muestren la interfaz de entrenamiento y supervisión una vez se cierra la interfaz de
    configuración
    """
    if gui.is_open == False:
        check_configuracion.stop()
        actualizar_entrenamiento.start()
        actualizar_supervision.start()
        gui_entrenamiento.showMaximized() #muestra en pantalla completa
        gui_supervision.show()
        gui.close()
        
    # else:
    #     # print('Controlando')

check_configuracion = QTimer() #Timer para control de que se cierre la interfaz de configuración
check_configuracion.setInterval(100)
check_configuracion.timeout.connect(iniciar_guis)
check_configuracion.start()

gui.show()
sys.exit(app.exec_())