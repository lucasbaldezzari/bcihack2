"""
Clase para analizar y graficar señales de EEG registradas durante sesiones de entrenamiento, 
calibración y online.

La clase EEGPlotter permite graficar señales de EEG con barras verticales indicando el inicio de los trials.

Se utiliza plotly ya que permite graficar e interactuar con los gráficos de una manera fluída a diferencia de matplotlib.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EEGPlotter:
    """
    Constructor de la clase EEGPlotter
    - eeg: numpy array de forma [canales, muestras] con los datos de EEG
    - fm: frecuencia de muestreo (en Hz)
    - paso: tamaño del paso (en segundos) para el slider
    - window_size: tamaño de la ventana (en segundos) para el slider
    - trial: lista con los números de los trials
    - trial_time: lista con los tiempos de inicio de los trials (en segundos)
    - y_max: límite superior del eje y
    - y_min: límite inferior del eje y
    """
    def __init__(self, eeg, fm, paso, window_size, trial=None, trial_time=None,
                 y_max=None, y_min=None):
        self.eeg = eeg
        self.fm = fm
        self.paso = paso
        self.window_size = window_size
        self.trial = trial if trial else []
        self.trial_time = trial_time if trial_time else []
        self.num_channels, self.num_samples = eeg.shape
        self.y_max = y_max
        self.y_min = y_min

        
        self.time_axis = self.getTimeAxis() # Creamos el eje de tiempo
        self.current_time = 0 # Iniciamos el tiempo actual en 0

        # Creamos subplots
        self.fig = make_subplots(rows=self.num_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                 subplot_titles=[''] * self.num_channels)

        # Calculamos la media para cada canal así centramos las señales en la gráfica
        self.middle_y_positions = [np.mean(eeg_channel) for eeg_channel in eeg]

        # Agregamos la info de cada canal y las barras verticales de los trials
        self.addTraces()
        
        # Seteamos el layout de la figura
        self.setLayout()

        self.quitarSpines()

        # Set slider configuration
        slider_steps = []
        for i in range(0, self.num_samples, int(self.paso * self.fm)):
            slider_steps.append({'args': [[self.time_axis[i]], {'frame': {'duration': 100, 'redraw': False},
                                                               'mode': 'immediate',
                                                               'transition': {'duration': 0}}],
                                 'label': f'{self.time_axis[i]:.2f}', 'method': 'animate'})

        self.fig.update_layout(
            sliders=[{
                'steps': slider_steps,
                'active': 0,
                'currentvalue': {'prefix': 'Time (s): '},
                'pad': {'t': 50}
            }]
        )

    def plot(self):
        # Show the interactive plot
        self.fig.show()

    def quitarSpines(self):
            # removemos las lineas de los subplots
        for i in range(self.num_channels):
            self.fig.update_xaxes(showline=True, linewidth=0, row=i+1, col=1)
            self.fig.update_yaxes(showline=False, linewidth=0, showticklabels=False, row=i+1, col=1)

            # Add the channel name to the left of each plot
            self.fig.update_yaxes(title_text=f'Canal {i+1}', title_standoff=0, row=i+1, col=1)

            # Set the y-axis limits if provided
            if self.y_min is not None and self.y_max is not None:
                self.fig.update_yaxes(range=[self.y_min, self.y_max], row=i+1, col=1)

    def getTimeAxis(self):
        return np.arange(0, self.num_samples) / self.fm
    
    def addTraces(self):
        for i in range(self.num_channels):
            self.fig.add_trace(go.Scatter(x=self.time_axis, y=self.eeg[i], showlegend=False), row=i+1, col=1)

            # Agregamos el número del trial a las señales de EEG
            trial_text = [f'Trial {trial_num}' for trial_num in self.trial]
            self.fig.add_trace(go.Scatter(x=self.trial_time, y=[self.middle_y_positions[i]] * len(self.trial),
                                text=trial_text,
                                mode='text', showlegend=False,
                                textposition='bottom center'), row=i+1, col=1)

            # Agregamos las barras verticales de los trials
            for trial_time in self.trial_time:
                if 0 <= trial_time <= self.num_samples / self.fm:
                    self.fig.add_shape(
                        type='line',
                        x0=trial_time,
                        x1=trial_time,
                        y0=0,
                        y1=1,
                        xref='x',
                        yref=f'y{i+1}',
                        line=dict(color='black', width=1, dash='dot'))

    def setLayout(self):
        self.fig.update_layout(
            title='Señal EEG',
            xaxis_title='Time (s)',
            yaxis_title='Amplitud $\mu V$',
            showlegend=False,
            height=self.num_channels * 200)  # Ajusta la altura de la figura según el número de canales)
        

if __name__ == "__main__":
    # Generar datos de prueba para EEG (4 canales x 2000 muestras)
    np.random.seed(42)
    num_channels = 3
    num_samples = 30000
    eeg = np.random.rand(num_channels, num_samples)

    # Frecuencia de muestreo (en Hz)
    fm = 250

    # Parámetros para el slider y el tamaño de ventana (en segundos)
    paso = 2
    window_size = 10

    # Números de los trials y tiempos de inicio de los trials (en segundos)
    trial = [1, 2, 3, 4, 5]
    trial_time = [5, 10, 15, 20, 25]

    # Crear la instancia de la clase EEGPlotter
    eeg_plotter = EEGPlotter(eeg, fm, paso, window_size, trial, trial_time,
                            y_max=1.5, y_min=-0.5)

    # Graficar la señal de EEG con las barras verticales indicando el inicio de los trials
    eeg_plotter.plot()