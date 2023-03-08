import numpy as np
from scipy.signal import hilbert, welch
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib import mlab

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """La clase para extraer características de desincronización relacionada con eventos (ERD) y sincronización relacionada con eventos (ERS) de señales EEG.
    La idea es usar la transformada de Hilbert para obtener la señal analítica y luego calcular la potencia de la señal en las bandas
    alfa y beta. La potencia en la banda alfa es la ERD y la potencia en la banda beta es la ERS.
    La clase se puede usar como un objeto de sklearn, por lo que se puede usar en un pipeline de sklearn.
    
    Comentario: Esta es una primera versión. Se debe probar otras estrategias para que la extracción de características sea más robusta.
    Por ejemplo, se puede implementar ICA para eliminar las componentes de ruido de la señal y luego aplicar la transformada de Hilbertm, o bien
    implementar CSP para extraer las componentes de interés de la señal y luego aplicar la transformada de Hilbert."""

    def __init__(self, method = "welch", sample_rate = 250., overlap = 0.5):
        """No se inicializan atributos.
        - method: método por el cual extraer la potencia
        - overlap: solapamiento en segundos"""

        self.method = method
        self.sample_rate = sample_rate
        self.overlap = overlap

    def fit(self, X = None, y=None):
        """No hace nada"""

        return self #este método siempre debe retornar self!

    def transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras, n_trials].
        
        Retorna: Un arreglo de numpy con las características de la señal. La forma del arreglo es [canales, power_sample, n_trials]"""
        
        if self.method == "welch":
            """Retorna la potencia de la señal en la forma [canales, power_sample, n_trials]"""
            #Aplicamos la transformada de Hilbert
            self.freqs, self.power = welch(signal, axis=1, nfft = signal.shape[1],
                                           noverlap=int(self.sample_rate*self.overlap)) #trnasformada de Welch
            self.freqs = self.freqs*self.sample_rate

            return self.power
        
        if self.method == "psd":
            #Calcula la PSD de la señal
            psdfunc = lambda x: mlab.psd(x, NFFT = signal.shape[1], Fs = self.sample_rate,
                                         noverlap=int(self.sample_rate*self.overlap))
            psd = np.apply_along_axis(psdfunc, 1, signal)
            self.freqs = psd[0,1,:,0]
            self.power = psd[:,0,:]

            return self.power

    def fit_transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras]."""

        self.fit()
        return self.transform(signal)

if __name__ == '__main__':

    with open("all_left_trials.npy", "rb") as f:
        signalLeft = np.load(f)

    with open("all_right_trials.npy", "rb") as f:
        signalRight = np.load(f)
    
    
    featureExtractor = FeatureExtractor(method="psd", sample_rate=100.) #instanciamos el extractor de características
    featuresleft = featureExtractor.fit_transform(signalLeft) #signal [n_channels, n_samples, n_trials]

    featureExtractor = FeatureExtractor(method="psd", sample_rate=100.) #instanciamos el extractor de características
    featuresright = featureExtractor.fit_transform(signalRight) #signal [n_channels, n_samples, n_trials]

    c3, cz, c4 = 26, 28, 30 #canales de interés

    import matplotlib.pyplot as plt
    plt.title("Potencia de la señal en la banda alfa y beta - Promedio trials - Clase $left$")
    plt.plot(featureExtractor.freqs, featuresleft[c3].mean(axis=1), label = "C3")
    plt.plot(featureExtractor.freqs, featuresleft[cz].mean(axis=1), label = "Cz")
    plt.plot(featureExtractor.freqs, featuresleft[c4].mean(axis=1), label = "C4")
    plt.xlim(1,30)
    plt.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    fig.suptitle("Potencia de la señal en las bandas alfa (mu) y beta - Promedio trials - Clase $left$ y $right$")
    ax[0].plot(featureExtractor.freqs, featuresleft[c3].mean(axis=1), label = "left")
    ax[0].plot(featureExtractor.freqs, featuresright[c3].mean(axis=1), label = "right")
    ax[0].set_title("C3")
    ax[0].grid()
    ax[0].set_xlim(1,30)
    ax[0].legend()
    ax[1].plot(featureExtractor.freqs, featuresleft[cz].mean(axis=1), label = "left")
    ax[1].plot(featureExtractor.freqs, featuresright[cz].mean(axis=1), label = "right")
    ax[1].set_title("Cz")
    ax[1].grid()
    ax[1].set_xlim(1,30)
    ax[1].legend()
    ax[2].plot(featureExtractor.freqs, featuresleft[c4].mean(axis=1), label = "left")
    ax[2].plot(featureExtractor.freqs, featuresright[c4].mean(axis=1), label = "right")
    ax[2].set_title("C4")
    ax[2].grid()
    ax[2].set_xlim(1,30)
    ax[2].legend()
    plt.show()

    with open("testing_left_features.npy", "wb") as f:
        np.save(f, featuresleft)