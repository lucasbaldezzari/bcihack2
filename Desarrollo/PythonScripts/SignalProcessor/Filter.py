import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, TransformerMixin

class Filter(BaseEstimator, TransformerMixin):
    """Clase para filtrar señales provenientes de la placa openBCI. Las señales llegan en un numpy array de la forma
    [canales, muestras]. La idea es aplicar un filtro pasa banda y un filtro notch a la señal a todo el array.
    La clase se puede usar como un objeto de sklearn, por lo que se puede usar en un pipeline de sklearn."""

    def __init__(self):
        """Inicializa el objeto con los parámetros de filtrado."""
        pass

    def fit(self, X = None, y=None, lowcut = 8.0, highcut = 30.0, notch_freq = 50.0, notch_width = 2.0, sample_rate = 250.0):
        """Creamos los filtros.
        -lowcut: Frecuencia de corte inferior del filtro pasa banda.
        -highcut: Frecuencia de corte superior del filtro pasa banda.
        -notch_freq: Frecuencia de corte del filtro notch.
        -notch_width: Ancho de banda del filtro notch.
        -sample_rate: Frecuencia de muestreo de la señal.
        -X: Señal de entrada. No se usa en este caso.
        -y: No se usa en este caso."""""

        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_width = notch_width
        self.sample_rate = sample_rate

        self.b, self.a = butter(5, [self.lowcut, self.highcut], btype='bandpass', fs=self.sample_rate)
        self.b_notch, self.a_notch = iirnotch(self.notch_freq, self.notch_width, self.sample_rate)

        return self #el método fit siempre debe retornar self!

    def transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un numpy array de la forma [canales, muestras]."""

        signal = np.subtract(signal, signal.mean(axis=0)) #restamos la media de cada muestra
        signal = filtfilt(self.b, self.a, signal, axis=1) #aplicamos el filtro pasa banda
        signal = filtfilt(self.b_notch, self.a_notch, signal, axis=1) #aplicamos el filtro notch
        return signal

def main():

    with open('testsignal.npy', 'rb') as f:
        signal = np.load(f)

    filtro = Filter()
    filtro.fit(lowcut=1.0, highcut=36.0, notch_freq=50.0, notch_width=2.0, sample_rate=250.0)
    signalFiltered = filtro.transform(signal)
    # signalFiltered = signalpros.fit_transform(signal,highcut=36.0, notch_freq=50.0, notch_width=2.0)

    ### Grafico para comparar señal original y señal filtrada
    import matplotlib.pyplot as plt
    plt.plot(signal[11,:])
    plt.plot(signalFiltered[11,:])
    plt.show()

    with open("testsignal_filtered.npy", "wb") as f:
        np.save(f, signalFiltered)


if __name__ == "__main__":
    main()