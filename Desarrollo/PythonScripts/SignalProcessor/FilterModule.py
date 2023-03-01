import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz, filtfilt, iirnotch, iirfilter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class SignalProcessor(BaseEstimator, TransformerMixin):
    """Clase para procesar señales provenientes del generador de señales (openBCI boards). Las señales que entran son un numpy array en la forma
    [canales, muestras]. La idea es aplicar un filtro pasa banda y un filtro notch a la señal a todo el array."""
    def __init__(self, sample_rate):
        """Inicializa el objeto con los parámetros de filtrado."""
        self.sample_rate = sample_rate

    def fit(self, X = None, y=None, lowcut = 1.0, highcut = 36.0, notch_freq = 50.0, notch_width = 2.0):
        """Creamos los filtros.
        -lowcut: Frecuencia de corte inferior del filtro pasa banda.
        -highcut: Frecuencia de corte superior del filtro pasa banda.
        -notch_freq: Frecuencia de corte del filtro notch.
        -notch_width: Ancho de banda del filtro notch.
        -X: Señal de entrada. No se usa en este caso.
        -y: No se usa en este caso."""""

        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_width = notch_width

        self.b, self.a = butter(5, [self.lowcut, self.highcut], btype='bandpass', fs=self.sample_rate)
        self.b_notch, self.a_notch = iirnotch(self.notch_freq, self.notch_width, self.sample_rate)

        return self #el método fit siempre debe retornar self!

    def transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un numpy array de la forma [canales, muestras]."""

        signal = filtfilt(self.b, self.a, signal, axis=1)
        signal = filtfilt(self.b_notch, self.a_notch, signal, axis=1)
        return signal

def main():

    with open('testsignal.npy', 'rb') as f:
        signal = np.load(f)

    # signal.shape

    signalpros = SignalProcessor(sample_rate=250)
    signalpros.fit(lowcut=1.0, highcut=36.0, notch_freq=50.0, notch_width=2.0)
    signalFiltered = signalpros.transform(signal)
    # signalFiltered = signalpros.fit_transform(signal,highcut=36.0, notch_freq=50.0, notch_width=2.0)


if __name__ == "__main__":
    main()