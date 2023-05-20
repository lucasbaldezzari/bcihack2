from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import pyqtgraph as pg
import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph import LabelItem

class SupervisionAPP(QDialog):
    """
    Interfaz gráfica para supervisar las sesiones
    """
    def __init__(self, clases:list, canales:list):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__), 'supervision.ui')
        uic.loadUi(ui_path, self)

        self.canales = canales

        pg.setConfigOptions(antialias=True)

        # create a new QGraphicsScene
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.graphics_window = pg.GraphicsLayoutWidget(title='EEG Plot', size=(950, 300))
        self.graphics_window.setBackground('w')
        scene.addWidget(self.graphics_window)

        # create a new QGraphicsScene
        scene2 = QGraphicsScene()
        self.graphicsBars.setScene(scene2)
        self.graphics_window2 = pg.GraphicsLayoutWidget(title='Bars', size=(400, 250))
        self.graphics_window2.setBackground('w')
        scene2.addWidget(self.graphics_window2)

        # create a new QGraphicsScene
        scene3 = QGraphicsScene()
        self.graphicsFFT.setScene(scene3)
        self.graphics_window3 = pg.GraphicsLayoutWidget(title='FFT', size=(500, 250))
        self.graphics_window3.setBackground('w')
        scene3.addWidget(self.graphics_window3)

        self.clases = clases

        self.sample_rate = 250.
        t_lenght = 10 # segundos de señal
        #Creo un numpyarray para hacer de buffer de datos que se mostrará en pantalla
        #El shape es (canales, tiempo)
        self.data = np.zeros((len(self.canales), int(self.sample_rate*t_lenght)))

        self.colores = ['#fb7e7b', '#ebcb5b', '#77aa99', '#581845', '#F7DC6F', '#F1C40F', '#9B59B6',
                        '#8E44AD', '#2980B9', '#2ECC71', '#27AE60', '#E67E22', '#D35400',
                        '#EC7063', '#D91E18', '#7D3C98']

        self._init_timeseries()
        self._init_barras()
        self._init_FFT()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()

        for i in range(len(self.canales)):
            p = self.graphics_window.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)

            p.setMenuEnabled('bottom', False)
            subtitle_label = LabelItem(f"Canal {i}", justify='center')
            subtitle_label.setText(f"Canal {i}", color='black', size='10pt', bold=True)
            subtitle_label.setPos(0.5, 1)  # Posición arriba y centrada
            subtitle_label.setParentItem(p.graphicsItem())  # Agregar el label al subplot

            p.showAxis('top', False)  # Ocultar el eje superior

            p.showAxis('bottom', False)

            self.plots.append(p)
            curve = p.plot(pen = self.colores[i])
            self.curves.append(curve)

    def update_plots(self, newData):

        samplesToRemove = newData.shape[1] #muestras a eliminar del buffer interno de datos
        ## giro el buffer interno de datos
        self.data = np.roll(self.data, -samplesToRemove, axis=1)
        ## reemplazo los ultimos datos del buffer interno con newData
        self.data[:, -samplesToRemove:] = newData

        for canal in range(len(self.canales)):
            self.curves[canal].setData(self.data[canal].tolist())

        self.update_FFT(self.data)

    def _init_barras(self):
        self.plots2 = list()
        self.bars = list()

        br = self.graphics_window2.addPlot(row=0, col=0)
        br.showAxis('left', True)
        br.setMenuEnabled('left', False)
        br.showAxis('bottom', True)
        br.setMenuEnabled('bottom', False)
        self.plots2.append(br)
        bottom_axis = br.getAxis('bottom')
        bottom_axis.setTicks([[(i, self.clases[i]) for i in range(len(self.clases))]])

        for i in range(len(self.clases)):
            bar = pg.BarGraphItem(x=[i], height=[0], width=0.5, brush=pg.mkBrush(self.colores[i]))
            br.addItem(bar)
            self.bars.append(bar)

    def update_bars(self, data = [0.5, 0.5]):
        for i, bar in enumerate(self.bars):
            bar.setOpts(height=data[i])

    def _init_FFT(self):
        self.plots3 = list()
        self.curves2 = list()

        p = self.graphics_window3.addPlot(row=0, col=0)
        p.showAxis('left', True)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', True)
        p.setMenuEnabled('bottom', False)
        self.plots3.append(p)

        for i in range(len(self.canales)):
            curve = p.plot(pen = self.colores[i])
            self.curves2.append(curve)

    def update_FFT(self, data):
        data = abs(np.fft.fft(data))[:,:100]
        for count, channel in enumerate(self.canales):
            self.curves2[count].setData(data[count].tolist())

    def actualizar_orden(self, texto:str):
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
    _ventana = SupervisionAPP(['AD', 'DP'], [1,2,3])

    _ventana.show()
    app.exec_()