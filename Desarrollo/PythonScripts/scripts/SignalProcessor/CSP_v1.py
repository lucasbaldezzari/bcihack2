"""Clase para aplicar el algoritmo CSP.
Uso:
  brede.eeg.csp [options]
Options:
  -h --help  Help

References
    ----------

    Christian Andreas Kothe,  Lecture 7.3 Common Spatial Patterns
    https://www.youtube.com/watch?v=zsOULC16USU

    Información acerca de "Common spatial pattern"
    https://en.wikipedia.org/wiki/Common_spatial_pattern

    ----------
"""

from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.linalg import eig
from sklearn import base

from FeatureExtractor import FeatureExtractor

class CSP(base.BaseEstimator, base.TransformerMixin):
    """CSP Class."""

    def __init__(self, n_components = None, method = "default", fileCSP = None):
        """Iniciamos la clase.
        -n_components: Número de componentes a extraer."""
        self.n_components = n_components
        self.method = method
        self.fileCSP = fileCSP #archivo para cargar un filtro o grupo de filtros ya entrenados

    @staticmethod
    def class_correlations(X):
        """Retornamos una lista con las matrices de correlación de cada clase."""
        correlationsClass = [np.corrcoef(X[0], rowvar=0), np.corrcoef(X[1], rowvar=0)]
        return correlationsClass

    @staticmethod
    def class_covariances(X):
        """Retorna una lista de matrices de covarianza de cada clase."""
        covarianzaClase = [np.cov(X[0], rowvar=0), np.cov(X[1], rowvar=0)]
        return covarianzaClase

    def fit(self, X, y=None):
        """
        Se usa fit para entrenar el modelo.
        La proyección se calcula cómo un problema de los valores propios o "eigenvalue problem" de la forma eig(covarianza_clase0, suma_covarianzas)
        Los pesos se ordenan de tal forma que los autovectores asociados con el mayor autovalor estén primero.

        Parametros
        ----------
        X : numpy.ndarray de la forma [n_channels, n_samples] (es nuestra matriz de datos que contiene el EEG)
        y : numpy.ndarray de la forma [n_classes] (es un vector con las etiquetas de las clases)

        Retorna
        -------
        self : CSP. El objeto self.
        """

        if self.method == "default":
            # Problema de los valores propios o "eigenvalue problem" en las covarianzas de las clases
            class_covariances = self.class_covariances(X)
            total_covariance = sum(class_covariances)
            eigenvalues, eigenvectors = eig(class_covariances[0], total_covariance)

            # Datos ordenados
            eigenvalues = np.real(eigenvalues)
            indices = np.argsort(-eigenvalues)
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]

            # Los parámetros del modelo
            if self.n_components is None:
                self.weights_ = eigenvectors
            else:
                self.weights_ = eigenvectors[:, :self.n_components]

        if self.method == "mne":
            pass

        return self

    def transform(self, X):
        """Proyectamos la matriz de datos con CSP.
        Parametros
        ----------
        X : numpy.ndarray de la forma [n_channels, n_samples] (es nuestra matriz de datos que contiene el EEG)

        Retorna
        -------
        X_new : numpy.ndarray de la forma [n_components, n_samples] (es nuestra matriz de EEG proyectada)
        """

        return X.dot(self.weights_)
    
if __name__ == "__main__":

    #Cargamos dos trials de señales correspondientes a dos clases diferentes. Una es cuando una persona imagina
    #que mueve su mano izquierda y la otra cuando imagina que mueve su mano derecha.
    #Cada trial tiene 59 canales.

    left = np.load("trialleft.npy", allow_pickle=True)
    right = np.load("righttrial.npy", allow_pickle=True)
    print(left.shape) #[n_channels, n_samples]
    print(right.shape) #[n_channels, n_samples]

    eegmatrix = np.array([left, right ])

    c3, cz, c4 = 26, 28, 30 #canales de interés
    
    #graficando la matriz de EEG para cada clase y un canal
    import matplotlib.pyplot as plt
    plt.title("EEG matrix antes de aplicar CSP")
    plt.plot(eegmatrix[0, 0, :], label = "left")
    plt.plot(eegmatrix[1, 0, :], label = "right")
    plt.legend()
    plt.show()

    #grafiando la matriz de EEG para cada clase y un canal
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle("EEG matrix antes de aplicar CSP C3, CZ, C4")
    ax1.plot(eegmatrix[0, c3, :], label = "left")
    ax1.plot(eegmatrix[1, c3, :], label = "right")
    ax1.legend()
    ax2.plot(eegmatrix[0, cz, :], label = "left")
    ax2.plot(eegmatrix[1, cz, :], label = "right")
    ax2.legend()
    ax3.plot(eegmatrix[0, c4, :], label = "left")
    ax3.plot(eegmatrix[1, c4, :], label = "right")
    ax3.legend()
    plt.show()

    #instanciamos la clase CSP
    csp = CSP()

    eegmatrix_csp = csp.fit_transform(eegmatrix)

    #graficando la matriz de EEG para cada clase y los tres canales de interes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle("EEG matrix después de aplicar CSP C3, CZ, C4")
    ax1.plot(eegmatrix_csp[0, c3, :], label = "left")
    ax1.plot(eegmatrix_csp[1, c3, :], label = "right")
    ax1.legend()
    ax2.plot(eegmatrix_csp[0, cz, :], label = "left")
    ax2.plot(eegmatrix_csp[1, cz, :], label = "right")
    ax2.legend()
    ax3.plot(eegmatrix_csp[0, c4, :], label = "left")
    ax3.plot(eegmatrix_csp[1, c4, :], label = "right")
    ax3.legend()
    plt.show()

    #calculamos la varianza de la matriz de EEG para cada clase y canal
    logvariance_no_csp = np.log(np.var(eegmatrix, axis=2))
    #ordenamos la varianza de la matriz de EEG para cada clase y canal
    logvariance_no_csp = np.sort(logvariance_no_csp, axis=1)
    
    #graficamos la varianza de la matriz de EEG para cada clase y canal
    plt.title("Logvariance de la matriz EEG antes de aplicar CSP")
    plt.bar(np.arange(0, 59), logvariance_no_csp[0, :], label = "left")
    plt.bar(np.arange(0, 59), logvariance_no_csp[1, :], label = "right")
    plt.legend()
    plt.show()

    #calculamos la varianza de la matriz de EEG luego de aplicar csp para cada clase y canal
    logvariance_csp = np.log(np.var(eegmatrix_csp, axis=2))
    index_sorted = np.argsort(np.log(np.var(eegmatrix_csp, axis=2)), axis=1)
    #ordenamos la varianza de la matriz de EEG luego de aplicar csp para cada clase y canal
    logvariance_csp = np.sort(logvariance_csp, axis=1)
    

    #graficamos la varianza de la matriz de EEG luego de aplicar csp para cada clase y canal
    plt.title("Logvariance de la matriz EEG después de aplicar CSP")
    plt.bar(np.arange(0, 59), logvariance_csp[0, :], label = "left")
    plt.bar(np.arange(0, 59), logvariance_csp[1, :], label = "right")
    plt.legend()
    plt.show()

    index_sorted.shape #[n_clases, n_canales]

    #Grafico para un canal y para cada clase
    plt.title("EEG matrix después de aplicar CSP")
    plt.plot(eegmatrix_csp[0, index_sorted[0,0], :], label = "left")
    plt.plot(eegmatrix_csp[1, index_sorted[1,-1], :], label = "right")
    plt.legend()
    plt.show()

    fe = FeatureExtractor(method="psd")

    left_features = fe.fit_transform(eegmatrix_csp[0, :, :])
    right_features = fe.fit_transform(eegmatrix_csp[1, :, :])

    #Grafico para un canal y para cada clase luego de aplicar feature extractor
    plt.title("EEG matrix después de aplicar CSP")
    plt.plot(left_features[index_sorted[0,0], :], label = "left")
    plt.plot(right_features[index_sorted[1,-1], :], label = "right")
    plt.legend()
    plt.show()

    ##pltting a scatter plot for the voltage of one class versus the voltage of the other class for one channel after applying CSP
    plt.title("Scatter plot de la matriz EEG antes de aplicar CSP")
    plt.scatter(eegmatrix[0, 0, :], eegmatrix[0, -1, :])
    plt.scatter(eegmatrix[1, 0, :], eegmatrix[1, -1, :])
    plt.legend(["primer componente", "segunda componente"])
    plt.show()

    #pltting a scatter plot for the voltage of one class versus the voltage of the other class for one channel before applying CSP
    plt.title("Scatter plot de la matriz EEG después de aplicar CSP")
    plt.scatter(eegmatrix_csp[0, 0, :], eegmatrix_csp[0, -1, :])
    plt.scatter(eegmatrix_csp[1, 0, :], eegmatrix_csp[1, -1, :])
    plt.legend(["primer componente", "segunda componente"])
    plt.show()

    fe = FeatureExtractor(method="psd")

    features_no_csp = fe.fit_transform(eegmatrix)
    features_csp = fe.fit_transform(eegmatrix_csp)

    #grafiando las features para los canales de interés antes del CSP
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle("Features antes de aplicar CSP - Canales C3, CZ, C4")
    ax1.plot(features_no_csp[0, c3, :], label = "left")
    ax1.plot(features_no_csp[1, c3, :], label = "right")
    ax1.legend()
    ax2.plot(features_no_csp[0, cz, :], label = "left")
    ax2.plot(features_no_csp[1, cz, :], label = "right")
    ax2.legend()
    ax3.plot(features_no_csp[0, c4, :], label = "left")
    ax3.plot(features_no_csp[1, c4, :], label = "right")
    ax3.legend()
    plt.show()

    #grafiando las features para los canales de interés luego del CSP
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle("Features antes de aplicar CSP - Canales C3, CZ, C4")
    ax1.plot(features_csp[0, c3, :], label = "left")
    ax1.plot(features_csp[1, c3, :], label = "right")
    ax1.legend()
    ax2.plot(features_csp[0, cz, :], label = "left")
    ax2.plot(features_csp[1, cz, :], label = "right")
    ax2.legend()
    ax3.plot(features_csp[0, c4, :], label = "left")
    ax3.plot(features_csp[1, c4, :], label = "right")
    ax3.legend()
    plt.show()



