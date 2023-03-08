import numpy as np
from scipy.signal import hilbert
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

    def __init__(self, method = "hilbert", sample_rate = 250., band = None):
        """No se inicializan atributos.
        - method: método por el cual extraer la potencia
        - band: lista con el ancho de banda de frecuencias a extraer. Ejemplo [8,30]. Si es None se retorna array completo"""

        self.method = method
        self.sample_rate = sample_rate
        self.band = band

    def fit(self, X = None, y=None):
        """No hace nada"""

        return self #este método siempre debe retornar self!

    def transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras].
        
        Retorna: Un arreglo de numpy con las características de la señal. La forma del arreglo es [canales, power_sample, trials]"""

        #NOTA: Debemos evaluar si implementamos un CSP para seleccionar los mejores canales y luego aplicamos la transformada de Hilbert

        if self.method == "hilbert":
            """Retorna la potencia de la señal en la forma [canales, power_sample, trials]"""
            #Aplicamos la transformada de Hilbert
            transformedSignal = hilbert(signal, axis=1) #trnasformada de Hilbert
            power = np.abs(transformedSignal)**2 #Calculamos la potencia de la señal
            self._alphaPower = power[:, 8:13]#.mean(axis=1) #Potencia media en la banda alfa
            self._betaPower = power[:, 13:30]#.mean(axis=1) #Potencia media en la banda beta
            features = np.hstack((self._alphaPower, self._betaPower)) #apilamos las características

            features = power

            return features
        
        elif self.method == "psd":
            #Calcula la PSD de la señal
            if self.band:
                psdfunc = lambda x: mlab.psd(x, NFFT = signal.shape[1], Fs = self.sample_rate)
                psd = np.apply_along_axis(psdfunc, 1, signal)
                self.freqs = psd[0,1,:]
                f1 = (self.freqs>=self.band[0])
                f2 = (self.freqs<=self.band[1])
                self.freqs = self.freqs[f1 & f2]
                self.power = psd[:,0,:]
                self.power = self.power[:,f1 & f2]
                return self.power
            else:
                psdfunc = lambda x: mlab.psd(x, NFFT = signal.shape[1], Fs = self.sample_rate)
                psd = np.apply_along_axis(psdfunc, 1, signal)
                self.power = psd[:,0,:]
                self.freqs = psd[0,1,:]
                return self.power

    def fit_transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras]."""

        self.fit()
        return self.transform(signal)

if __name__ == '__main__':
    with open("testsignal_filtered.npy", "rb") as f:
        signal = np.load(f)
    
    featureExtractor = FeatureExtractor(method="psd", sample_rate=250, band = None) #instanciamos el extractor de características

    features = featureExtractor.fit_transform(signal) #signal [n_channels, n_samples]
    features.shape

    featureExtractor.freqs.shape
    

    import matplotlib.pyplot as plt
    plt.plot(featureExtractor.freqs, features[0,:])
    plt.show()

    filtro = featureExtractor.freqs>=8
    filtro2 = featureExtractor.freqs<=30

    features[:,(filtro)&(filtro2)]




    with open("testing_features.npy", "wb") as f:
        np.save(f, features)