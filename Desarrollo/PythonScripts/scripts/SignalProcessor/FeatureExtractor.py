import numpy as np
from scipy.signal import hilbert, welch
from sklearn.base import BaseEstimator, TransformerMixin
# from matplotlib import mlab

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """La clase para extraer características de desincronización relacionada con eventos (ERD) y sincronización relacionada con eventos (ERS) de señales EEG.
    - La clase puede extraer la envolvente de la señal a partir de Hilber, o,
    - La clase puede extraer la potencia de la señal a partir de la transformada de Welch.

    La clase se comporta como un transformer de sklearn, por lo que puede ser usada en un pipeline de sklearn.
    """

    def __init__(self, method = "welch", sample_rate = 250., axisToCompute = 2):
        """No se inicializan atributos.
        - method: método por el cual extraer las caracerísticas. Puede ser welch o hilbert.
        - sample_rate: frecuencia de muestreo de la señal.
        - axisToCompute: eje a lo largo del cual se calculará la transformada."""

        self.method = method
        self.sample_rate = sample_rate
        self.axisToCompute = axisToCompute

    def fit(self, X = None, y=None):
        """No hace nada"""

        return self #este método siempre debe retornar self!

    def transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [n_trials, canales, muestras].
        
        Retorna: Un arreglo de numpy con las características de la señal. La forma del arreglo es [canales, power_sample, n_trials]"""
        
        if self.method == "welch":
            """Retorna la potencia de la señal en la forma [n_trials, canales, power_samples]"""
            self.freqs, self.power = welch(signal, axis=self.axisToCompute) #trnasformada de Welch
            
            self.freqs = self.freqs*self.sample_rate
            return self.power
        
        if self.method == "hilbert":
            """Retorna la potencia de la señal en la forma [n_trials, canales, power_samples]"""
            analyticSignal = hilbert(signal, axis=self.axisToCompute)
            self.envolvente = np.abs(analyticSignal) #envolvente de la señal analítica
            return self.envolvente
        
    def fit_transform(self, signal, y = None):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras]."""

        self.fit()
        return self.transform(signal)

if __name__ == '__main__':

    with open("testData/all_left_trials.npy", "rb") as f:
        signalLeft = np.load(f)

    with open("testData/all_right_trials.npy", "rb") as f:
        signalRight = np.load(f)
    

    ## Extraemos envolventes de las señals
    featureExtractor = FeatureExtractor(method="hilbert", sample_rate=100.) #instanciamos el extractor de características
    featuresleft = featureExtractor.fit_transform(signalLeft) #signal [n_channels, n_samples, n_trials]

    featureExtractor = FeatureExtractor(method="hilbert", sample_rate=100.) #instanciamos el extractor de características
    featuresright = featureExtractor.fit_transform(signalRight) #signal [n_channels, n_samples, n_trials]

    c3, cz, c4 = 26, 28, 30 #canales de interés

    t1 = -0.5
    t2 = 2.5
    sample_rate = 100.
    timeline = np.arange(t1, t2, 1/sample_rate)
    timeline.shape

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    fig.suptitle("Envolvente de la señal - Promedio trials - Clases $left$ y $right$")
    ax[0].plot(timeline, featuresleft.mean(axis=0)[c3], label = "left")
    ax[0].plot(timeline, featuresright.mean(axis = 0)[c3], label = "right")
    ax[0].set_title("C3")
    ax[0].set_xlabel("Tiempo (s)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(timeline, featuresleft.mean(axis=0)[cz], label = "left")
    ax[1].plot(timeline, featuresright.mean(axis = 0)[cz], label = "right")
    ax[1].set_title("Cz")
    ax[1].set_xlabel("Tiempo (s)")
    ax[1].legend()
    ax[1].grid()
    ax[2].plot(timeline, featuresleft.mean(axis=0)[c4], label = "left")
    ax[2].plot(timeline, featuresright.mean(axis = 0)[c4], label = "right")
    ax[2].set_title("C4")
    ax[2].set_xlabel("Tiempo (s)")
    ax[2].legend()
    ax[2].grid()
    plt.show()


    ## Extraemos potencia de las señales
    featureExtractor = FeatureExtractor(method="welch", sample_rate=100.) #instanciamos el extractor de características
    featuresleft = featureExtractor.fit_transform(signalLeft) #signal [n_channels, n_samples, n_trials]

    featureExtractor = FeatureExtractor(method="welch", sample_rate=100.) #instanciamos el extractor de características
    featuresright = featureExtractor.fit_transform(signalRight) #signal [n_channels, n_samples, n_trials]


    fig, ax = plt.subplots(1,3, figsize=(10,5))
    fig.suptitle("Potencia de la señal en las bandas alfa (mu) y beta - Promedio trials - Clases $left$ y $right$")
    ax[0].plot(featureExtractor.freqs, featuresleft.mean(axis=0)[c3], label = "left")
    ax[0].plot(featureExtractor.freqs, featuresright.mean(axis=0)[c3], label = "right")
    ax[0].set_title("C3")
    ax[0].grid()
    ax[0].set_xlim(8,30)
    ax[0].legend()
    ax[1].plot(featureExtractor.freqs, featuresleft.mean(axis=0)[cz], label = "left")
    ax[1].plot(featureExtractor.freqs, featuresright.mean(axis=0)[cz], label = "right")
    ax[1].set_title("Cz")
    ax[1].grid()
    ax[1].set_xlim(8,30)
    ax[1].legend()
    ax[2].plot(featureExtractor.freqs, featuresleft.mean(axis=0)[c4], label = "left")
    ax[2].plot(featureExtractor.freqs, featuresright.mean(axis=0)[c4], label = "right")
    ax[2].set_title("C4")
    ax[2].grid()
    ax[2].set_xlim(8,30)
    ax[2].legend()
    plt.show()

    with open("testing_left_features.npy", "wb") as f:
        np.save(f, featuresleft)