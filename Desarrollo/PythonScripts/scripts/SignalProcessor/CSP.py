"""Clase para aplicar el algoritmo CSP.
Uso:
  brede.eeg.csp [options]
Options:
  -h --help  Help

References
    ----------

    Christian Andreas Kothe,  Lecture 7.3 Common Spatial Patterns
    https://www.youtube.com/watch?v=zsOULC16USU

    https://github.com/fnielsen/brede/blob/4ae9c6c1bea5c00751606bb9b9421fa46fd1e9bc/brede/eeg/csp.py

    Información acerca de "Common spatial pattern"
    https://en.wikipedia.org/wiki/Common_spatial_pattern

    ----------
"""

from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.linalg import eig
from sklearn import base

import FeatureExtractor

class CSP(base.BaseEstimator, base.TransformerMixin):
    """CSP Class."""

    def __init__(self, n_components = None):
        """Iniciamos la clase.
        -n_components: Número de componentes a extraer."""
        self.n_components = n_components

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
    #Cada trial tiene 59 canales y n muestras

    left = np.load("trialleft.npy", allow_pickle=True)
    right = np.load("trialright.npy", allow_pickle=True)
    print(left.shape) #[n_channels, n_samples]
    print(right.shape) #[n_channels, n_samples]

    eegmatrix = np.array([left, right])
    
    #plotting the eeegmatrix for each class and one channel
    import matplotlib.pyplot as plt
    plt.title("EEG matrix antes de aplicar CSP")
    plt.plot(eegmatrix[0, 0, :], label = "left")
    plt.plot(eegmatrix[1, 0, :], label = "right")
    plt.legend()
    plt.show()

    csp = CSP()

    eegmatrix_csp = csp.fit_transform(eegmatrix)

    #plotting the eeegmatrix_csp for each class and one channel
    plt.title("EEG matrix después de aplicar CSP")
    plt.plot(eegmatrix_csp[0, 0, :], label = "left")
    plt.plot(eegmatrix_csp[1, 0, :], label = "right")
    plt.legend()
    plt.show()