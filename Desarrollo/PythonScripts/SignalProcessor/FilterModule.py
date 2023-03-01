import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz, filtfilt, iirnotch, iirfilter

class SignalProcessor():
    """Clase para procesar señales provenientes del generador de señales (openBCI boards). Las señales que entran son un numpy array en la forma
    [canales, muestras]. La idea es aplicar un filtro pasa banda y un filtro notch a la señal a todo el array."""
    def __init__(self, sample_rate, lowcut, highcut, notch_freq, notch_width):
        """Inicializa el objeto con los parámetros de filtrado."""
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_width = notch_width

        self.b, self.a = self.butter_bandpass(lowcut, highcut, sample_rate, order=5)
        self.b_notch, self.a_notch = self.iirnotch(notch_freq, notch_width, sample_rate)

    def filterSignal(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: numpy array de la forma [canales, muestras]."""
        signalFiltered = filtfilt(self.b, self.a, signal, axis=1)
        signalFiltered = filtfilt(self.b_notch, self.a_notch, signal, axis=1)
        return signalFiltered