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

    def __init__(self, method = "hilbert"):
        """No se inicializan atributos."""

        self.method = method
        pass

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
            alphaPower = power[:, 8:13]#.mean(axis=1) #Potencia media en la banda alfa
            betaPower = power[:, 13:30]#.mean(axis=1) #Potencia media en la banda beta
            features = np.hstack((alphaPower, betaPower)) #apilamos las características

            features = power

            return features
        
        elif self.method == "psd":
            #Calcula la PSD de la señal
            psd = np.apply_along_axis(mlab.psd, 1, signal)
            # features = np.vstack((psd[:, 8:13], psd[:, 13:30]))
            return psd[:,0,:,:]

    def fit_transform(self, signal):
        """Función para aplicar los filtros a la señal.
        -signal: Es la señal en un arreglo de numpy de la forma [canales, muestras]."""

        self.fit()
        return self.transform(signal)

def main():

    with open("testsignal_filtered.npy", "rb") as f:
        signal = np.load(f)

    featureExtractor = FeatureExtractor() #instanciamos el extractor de características

    features = featureExtractor.fit_transform(signal)

    with open("testing_features.npy", "wb") as f:
        np.save(f, features)


if __name__ == '__main__':
    main()