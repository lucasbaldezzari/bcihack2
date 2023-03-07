"""Clase para aplicar CSP

La clase hace uso de mne.decoding.csp (https://mne.tools/stable/generated/mne.decoding.CSP.html)

Si bien esta clase puede parecer redundante, en realidad se agregan algunos métodos para el correcto funcionamiento
del pipeline del Hackathon 2.

References
    ----------

    Christian Andreas Kothe,  Lecture 7.3 Common Spatial Patterns
    https://www.youtube.com/watch?v=zsOULC16USU

    Información acerca de "Common spatial pattern"
    https://en.wikipedia.org/wiki/Common_spatial_pattern


    Optimizing Spatial filters for Robust EEG Single-Trial Analysis
    https://ieeexplore.ieee.org/document/4408441

    https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-decoding-csp-eeg-py

    ----------
"""

import numpy as np
import pickle
from sklearn import base

from mne.decoding import CSP

class commonSpatialPattern(base.BaseEstimator, base.TransformerMixin):
    """Clase CSP. El código base usa mne.decoding.csp. Se agregan algunos métodos para funcionalidad 
    del sistema correspondiente al segundo hackathon"""

    def __init__(self, n_components=4, reg=None, log=None, cov_est='concat',
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=None,
                 component_order='mutual_info') -> None:
        """Constructor de clase
        
        Para información de los parámetros reg, log y transform_into referenciarse a mne.decoding.csp"""
       
        self.n_components = n_components
        self.rank = rank
        self.reg = reg
        self.cov_est = cov_est
        self.log = log
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.transform_into = transform_into

        self._csp = CSP(n_components = self.n_components, reg=reg, log=log, cov_est=cov_est,
                 transform_into=transform_into, norm_trace=norm_trace, cov_method_params=cov_method_params,
                 rank=rank, component_order=component_order)

    @staticmethod
    def loadCSP(self, filename = "csp_model.pkl"):
        """Cargamos un modelo ya entrenado a partir de un archivo .pickle"""
        
        with open(filename, "rb") as f:
            self._csp = pickle.load(f)

    def fit(self, X, y):
        """Determina los filtros espaciales a partir de las épocas dentro de nuestros
        
        - X: numpy.array de la forma [n_epochs, n_channels, n_samples]
        - y: array con las etiquetas de clase. El array es de la forma [n_epochs].
        """

        self._csp.fit(X,y)

        return self #El método fit siempre retorna self
    
    def transform(self, X, y = None):
        return self._csp.transform(X)
        

    def saveCSPFilters(self, filname = "csp_model.pkl"):
        """Guardamos el modelo"""
        with open(filname, 'wb') as f:
            pickle.dump(self._csp, f)

    def __str__(self):
    #Do whatever you want here
            return "Name: {0}\tGenders: {1} Country: {2} ".format(self.name,self.genders,self.country)



if __name__ == "__main__":

    left = np.load("all_left_trials.npy", allow_pickle=True)
    right = np.load("all_right_trials.npy", allow_pickle=True)
    print(left.shape) #[n_channels, n_samples, ntrials]
    print(right.shape) #[n_channels, n_samples, ntrials]

    c3, cz, c4 = 26, 28, 30 #canales de interés

    csp = commonSpatialPattern(n_components = 4)
    # print(csp._csp)

    #los datos deben pasarse al csp de la forma [n_epochs, n_channels, n_samples]
    #los datos que cargamos en eegmatrix están de la forma [n_clases, n_channels, n_samples]
    #debemos cambiar los datos a la forma que necesita la clase CSP.

    left = np.swapaxes(left, 2,1).swapaxes(1,0)
    right = np.swapaxes(right, 2,1).swapaxes(1,0)
    leftright = left
    print(left.shape) #[ntrials, n_channels, n_samples]
    print(right.shape) #[ntrials, n_channels, n_samples]

    #Contactemos en un sólo array
    eegmatrix = np.concatenate((left,right)) #importante el orden con el que concatenamos
    print(eegmatrix.shape) #[n_epochs, n_channels, n_samples]

    #IMPORTANTE: DEBEMOS PRESTAR ESPECIAL ANTENCIÓN AL ORDEN DE LAS CLASES YA QUE EL CSP NOS DEVOLVERÁ UN SET DE FILTROS ESPACIALES
    #CADA FILTRO ESPACIAL SE CORRESPONDE A UNA DE LAS CLASES EN EL ORDEN CON QUE SE ENTRENA EL CSP

    class_info = {1: "left", 2: "right"} #diccionario para identificar clases. El orden se corresponde con lo que hay eneegmatrix

    #En este ejemplo tenemos los trials ordenados, es decir, tenemos 100 trials para la clase left y 100 para la clase right.
    #Por lo tanto, las etiquetas que crearemos estarán ordenadas.
    n_trials = left.shape[0]
    n_epochs = eegmatrix.shape[0]
    labels = np.array([np.full(n_trials, label) for label in class_info.keys()]).reshape(n_epochs)
    print(labels.shape)
    print(labels)

    csp.fit(eegmatrix, labels) #entrenamos csp

    #projectamos un sólo trial.
    left_csp = csp.transform(left)
    right_csp = csp.transform(right)

    right_csp.shape

    ## COMPARANDO DATOS ANTES Y DESPUÉS DE APLICAR CSP
    import matplotlib.pyplot as plt
    # plt.scatter(right[:,0], right[:,-1], label = "right sin csp" )
    plt.scatter(right_csp[:,0], right_csp[:,-1], label = "right_csp")
    plt.scatter(left_csp[:,0], left_csp[:,-1], label = "left_csp")
    plt.show()

    


