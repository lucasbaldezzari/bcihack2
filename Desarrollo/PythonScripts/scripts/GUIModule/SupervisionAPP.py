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
    def __init__(self, clases:list, canales:list, umbral_clasificacion = 75):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__), 'supervision.ui')
        uic.loadUi(ui_path, self)

        self.canales = canales
        self.clases = clases

        self.umbral_calsificacion = umbral_clasificacion #% umbral de clasificación

        self.sample_rate = 250.
        self.t_lenght = 10 # segundos de señal
        #Creo un numpyarray para hacer de buffer de datos que se mostrará en pantalla
        #El shape es (canales, tiempo)
        self.data = np.zeros((len(self.canales), int(self.sample_rate*self.t_lenght)))

        #creo eje tiempo
        self.tline = np.linspace(0, self.t_lenght, int(self.sample_rate*self.t_lenght))

        #creo eje frecuencia
        self.fline = np.linspace(0, self.sample_rate/2, int(self.sample_rate*self.t_lenght/2))
        #frecuencias mínima y máxima a graficar
        self.fmin = 5
        self.fmax = 30

        pg.setConfigOptions(antialias=True)

        # new QGraphicsScene for timeseries EEG data
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.graphics_window = pg.GraphicsLayoutWidget(title='EEG Plot', size=(950, 390))
        self.graphics_window.setBackground('w')
        scene.addWidget(self.graphics_window)

        # new QGraphicsScene for probabilties bars
        scene2 = QGraphicsScene()
        self.graphicsBars.setScene(scene2)
        self.graphics_window2 = pg.GraphicsLayoutWidget(title='Bars', size=(400, 250))
        self.graphics_window2.setBackground('w')
        scene2.addWidget(self.graphics_window2)

        # new QGraphicsScene for FFT data
        scene3 = QGraphicsScene()
        self.graphicsFFT.setScene(scene3)
        self.graphics_window3 = pg.GraphicsLayoutWidget(title='FFT', size=(500, 250))
        self.graphics_window3.setBackground('w')
        scene3.addWidget(self.graphics_window3)

        #8 colores para los canales de EEG
        self.colores_eeg = ['#fb7e7b', '#ebcb5b', '#77aa99', '#581845', '#F7DC6F', '#F1C40F', '#9B59B6','#8E44AD']

        #colores para las barras de probabilidad
        self.colores_barras = ['#8199c8', '#b58fbb', '#77aa99', '#edcf5b', '#fa7f7c', '#F1C40F', '#9B59B6','#8E44AD']

        self._init_timeseries()
        self._init_barras()
        self._init_FFT()

        self.update_bars([0.0 for i in range(len(self.clases))])

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()

        for i in range(len(self.canales)):
            p = self.graphics_window.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', True)
            p.showGrid(x=True, y=True)

            ax0 = p.getAxis('left') #para manipular el axis izquierdo (eje y)
            ax0.setStyle(showValues=False)
            ax0.setLabel(f"C{self.canales[i]}", color=self.colores_eeg[i], size='14pt', bold=True)

            ## Creo eje inferior
            ax1 = p.getAxis('bottom') #para manipular el eje inferior (eje x)
            ax1.setStyle(showValues=True)
            ax1.setTickFont(QFont('Arial', 8))
            ax1.setRange(0, self.t_lenght)
            
            p.showAxis('top', False)  # Ocultar el eje superior
            p.showAxis('bottom', True)

            # p.addLegend(size=(2,0), offset=(-0.5,-0.1))# agrego leyenda por cada gráfico
            curve = p.plot(pen = self.colores_eeg[i], name = f'Canal {i}')
            self.curves.append(curve)

    def _init_FFT(self):
        self.plots3 = list()
        self.curves2 = list()

        p = self.graphics_window3.addPlot(row=0, col=0)
        p.showAxis('left', True)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', True)
        p.setMenuEnabled('bottom', False)
        p.showGrid(x=True, y=True)

        ax1 = p.getAxis('left') #para manipular el eje inferior (eje x)
        ax1.setStyle(showValues=False)
        ax1.setLabel(f"Amplitud (uv)", color="k", size='14pt', bold=True)

        self.plots3.append(p)

        for i in range(len(self.canales)):
            curve = p.plot(pen = self.colores_eeg[i])
            self.curves2.append(curve)

    def update_plots(self, newData):

        samplesToRemove = newData.shape[1] #muestras a eliminar del buffer interno de datos
        ## giro el buffer interno de datos
        self.data = np.roll(self.data, -samplesToRemove, axis=1)
        ## reemplazo los ultimos datos del buffer interno con newData
        self.data[:, -samplesToRemove:] = newData

        for canal in range(len(self.canales)):
            self.curves[canal].setData(self.tline,self.data[canal])

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
        bottom_axis = br.getAxis('bottom') #eje inferior
        bottom_axis.setTicks([[(i, self.clases[i]) for i in range(len(self.clases))]])

        ## Manipulando eje izquierdo
        lef_axis = br.getAxis('left')
        lef_axis.setStyle(showValues=True)
        lef_axis.setPen('k')
        lef_axis.setTickFont(QFont('Arial', 10))
        lef_axis.setRange(0, 100)

        ## Agrego una línea horizontal en 100%
        self.hline100 = pg.InfiniteLine(pos=100, angle=0, movable=False, pen='k')
        br.addItem(self.hline100)

        ## Agrego una lina para el umbral
        hline_umbral = pg.InfiniteLine(pos=self.umbral_calsificacion, angle=0, movable=False, pen='k')
        hline_umbral.setPen(pg.mkPen('#548711', width=1, style=Qt.DashLine))
        br.addItem(hline_umbral)

        for i in range(len(self.clases)):
            bar = pg.BarGraphItem(x=[i], height=[0], width=0.8, brush=pg.mkBrush(self.colores_barras[i]))
            br.addItem(bar)
            self.bars.append(bar)

    def update_bars(self, data = [0.5, 0.5]):
        for i, bar in enumerate(self.bars):
            bar.setOpts(height=round(data[i]*100,2))

    def update_FFT(self, data):
        """Calcula la FFT de los datos y los grafica en self.fmin y self.fmax"""

        #tomo los datos de la fft y los grafico en self.fmin y self.fmax
        data = abs(np.fft.fft(data).real)[:,int(self.fmin*self.t_lenght):int(self.fmax*self.t_lenght)]

        fline = np.linspace(self.fmin, self.fmax, data.shape[1])

        for canal in range(len(self.canales)):
            self.curves2[canal].setData(fline, data[canal])

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