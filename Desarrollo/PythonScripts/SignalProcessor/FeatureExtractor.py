import numpy as np
from scipy.signal import hilbert, filtfilt, butter
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """La clase para extraer características de desincronización relacionada con eventos (ERD) y sincronización relacionada con eventos (ERS) de señales EEG.
    La idea es usar la transformada de Hilbert para obtener la señal analítica y luego calcular la potencia de la señal en las bandas
    alfa y beta. La potencia en la banda alfa es la ERD y la potencia en la banda beta es la ERS.
    La clase se puede usar como un objeto de sklearn, por lo que se puede usar en un pipeline de sklearn.
    
    Comentario: Esta es una primera versión. Se debe probar otras estrategias para que la extracción de características sea más robusta.
    Por ejemplo, se puede implementar ICA para eliminar las componentes de ruido de la señal y luego aplicar la transformada de Hilbertm, o bien
    implementar CSP para extraer las componentes de interés de la señal y luego aplicar la transformada de Hilbert."""

    def __init__(self):
        """No se inicializan atributos."""
        pass

    def fit(self, X = None, y=None):
        """No hace nada"""

        return self #este método siempre debe retornar self!

    def transform(self, signal):
        """Function to apply the filters to the signal.
        -signal: It is the signal in a numpy array of the form [channels, samples]."""

        #NOTA: Debemos evaluar si implementamos un CSP para seleccionar los mejores canales y luego aplicamos la transformada de Hilbert

        #Aplicamos la transformada de Hilbert
        analytic_signal = hilbert(signal, axis=1) #trnasformada de Hilbert
        power = np.abs(analytic_signal)**2 #Calculamos la potencia de la señal
        power_alpha = power[:, 8:13].mean(axis=1) #Potencia media en la banda alfa
        power_beta = power[:, 13:30].mean(axis=1) #Potencia media en la banda beta
        features = np.vstack((power_alpha, power_beta)).T #apilamos las características

        return features

    def csp_filter(self, signal):
        """TODO: Implementar el filtro CSP para seleccionar los mejores canales y luego aplicar la transformada de Hilbert."""
        pass