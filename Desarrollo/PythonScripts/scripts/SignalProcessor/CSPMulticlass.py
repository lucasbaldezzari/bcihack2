"""
Clase para generar y/o aplicar filtros CSP para múltiples movimientos (clases). Se puede seleecionar entre el método "one vs one" o "one vs all. En el primer cao
tendremos K(K-1)/2 filtros (donde K es la cantidad de movimientos/clases) y en el segundo se tendrán tantos filtros como clases/movimientos tenga la BCI.

Esta clase hará uso de la clase mne.decoding.csp para generar los filtros CSP. Para más información sobre esta clase, ver la documentación de mne.

1) El constructor de CSPMulticlass recibe los parámetros que recibe la clase mne.decoding.csp. Estos parámetros se utilizarán para entrenar cada clasificador
2) El método fit recibe los datos de entrenamiento y las etiquetas de clase. Estos datos se utilizarán para entrenar cada filtro CSP. El método fit retorna self
3) El método transform recibe los datos a transformar y retorna los datos transformados. Este método no se utiliza para entrenar los filtros CSP.


 filters_ attribute only returns the top m CSP filters, meaning the m filters with the largest eigenvalues. These m filters are the ones that capture the most discriminative information between the two classes.

When transform_into='csp_space', the transform() method applies these top m filters to the input data, resulting in a new feature space that is projected onto the CSP components. 

IMPORTANTE: Debemos tener en cuenta que el método mne.decoding.csp posee un atributo filters_. El mismo posee la matriz de filtros espaciales luego de aplicar
fit(). La clase CSP devuelve sólo los m filtros con las mayores autovalores. Estos m filtros son los que capturan la información discriminante más importante
entre las dos clases. Cuando transform_into='csp_space', el método transform() aplica estos m filtros a los datos de entrada, resultando en un nuevo espacio de
características que está proyectado sobre los componentes CSP. A diferencia de otras implementaciones, el método de mne sólo devuelve los m filtros con las mayores
autovalores. Esto significa que si queremos utilizar además los m filtros correspondientes a los m menores autovalores, debemos hacerlo manualmente utilizando
los atributos filters_, eigenvalues_ y eigenvectors_ de la clase mne.decoding.csp, ejemplo:


eigenvalues = csp.eigenvalues_
eigenvectors = csp.filters_

sorted_indices = np.argsort(eigenvalues)[::-1]  # ordenamos los índices
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

m = 3
top_filters = sorted_eigenvectors[:, :m]
bottom_filters = sorted_eigenvectors[:, -m:]
"""

import numpy as np
import pickle
from sklearn import base
import itertools

from mne.decoding import CSP

class CSPMulticlass(base.BaseEstimator, base.TransformerMixin):
    """TODO: Documentación de clase"""

    def __init__(self, method = "ovo", n_classes = 5, n_components=2, reg=None, log=None, cov_est='concat',
                 transform_into='csp_space', norm_trace=False,
                 cov_method_params=None, rank=None,
                 component_order='mutual_info') -> None:
        """Constructor de clase

        Parámetros:
        ----------------
        - method: str. Es el método a utilizar para entrenar los filtros CSP. Puede ser "ovo" u "ova". Por defecto es "ovo"
        - n_classes: int. Es la cantidad de clases que se pretende discriminar con CSP. Por defecto es 5
        - n_components: int. Es la cantidad de componentes a extraer. Por defecto es 2
        - reg: float. Es el parámetro de regularización. Por defecto es None
        - log: bool. Si es True, se aplica logaritmo a la matriz de covarianza. Por defecto es None
        - cov_est: str. Es el método a utilizar para estimar la matriz de covarianza. Puede ser "concat", "epoch" o "auto". Por defecto es "concat"
        - transform_into: str. Es el método a utilizar para transformar los datos. Puede ser "average_power" o "csp_space". Por defecto es "csp_space"
        - norm_trace: bool. Si es True, se normaliza la traza de la matriz de covarianza. Por defecto es False
        - cov_method_params: dict. Es un diccionario con los parámetros para el método de estimación de covarianza. Por defecto es None
        - rank: int. Es el rango de la matriz de covarianza. Por defecto es None
        - component_order: str. Es el método a utilizar para ordenar los componentes. Puede ser "mutual_info" o "shuffle". Por defecto es "mutual_info"

        Para más información sobre los parámetros (excepto method y n_classes), ver la documentación de mne.decoding.csp"""


        self.method = method
        self.n_classes = n_classes
        self.n_components = n_components
        self.rank = rank
        self.reg = reg
        self.cov_est = cov_est
        self.log = log
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = component_order
        self.transform_into = transform_into

        #lista de filtros CSP
        if method == "ovo":
            self.csplist = [CSP(n_components = self.n_components, reg=reg, log=log, cov_est=cov_est,
                 transform_into=transform_into, norm_trace=norm_trace, cov_method_params=cov_method_params,
                 rank=rank, component_order=component_order) for i in range(int((n_classes*(n_classes-1))/2))]
            
        if method == "ova":
            self.csplist = [CSP(n_components = self.n_components, reg=reg, log=log, cov_est=cov_est,
                 transform_into=transform_into, norm_trace=norm_trace, cov_method_params=cov_method_params,
                 rank=rank, component_order=component_order) for i in range(n_classes)]
            
    def saveCSPList(self, filename, folder = "filtrosCSP"):
        """Método para guardar los filtros CSP en un archivo pickle

        Parámetros:
        ----------------
        - filename: str. Es el nombre del archivo donde se guardarán los filtros CSP
        - folder: str. Es el nombre de la carpeta donde se guardarán los filtros CSP. Por defecto es "filtrosCSP"

        Retorna:
        ----------------
        - None"""

        #guardamos en un archivo pickle
        with open(folder + "/" + filename, "wb") as f:
            pickle.dump(self.csplist, f)
        
    @staticmethod #método estático para poder cargar filtros CSP sin instanciar la clase
    def loadCSPList(self, filename, folder = "filtrosCSP"):
        """Método para cargar los filtros CSP de un archivo pickle

        Parámetros:
        ----------------
        - filename: str. Es el nombre del archivo donde se guardarán los filtros CSP
        - folder: str. Es el nombre de la carpeta donde se guardarán los filtros CSP. Por defecto es "filtrosCSP"

        Retorna:
        ----------------
        - None"""

        with open(folder + "/" + filename, "rb") as f:
            self.csplist = pickle.load(f)


    def fit(self, X, y):
        """Método para entrenar los filtros CSP

        - Las señales de EEG vienen en el formato (n_trials, n_channels, n_samples).
        - Si el método es "ovo", se entrena un filtro CSP para cada combinación de clases. Clase1 vs Clase2, Clase1vsClase3, etc.
        - Si el método es "ova", se entrena un filtro CSP para cada clase. Donde se toma clase1 vs todas las demás clases,
        clase 2 vs todas las demás clases, etc.
        - Los filtros a entrenar se encuentran dentro de la lista self.csplist

        Parámetros:
        ----------------
        - X: ndarray. Es un arreglo de numpy con los datos de entrenamiento. Tiene el formato (n_trials, n_channels, n_samples)
        - y: ndarray. Es un arreglo de numpy con las etiquetas de clase. Tiene el formato (n_trials,)

        Retorna:
        ----------------
        - self"""

        if self.method == "ovo":
            classlist = np.unique(y)
            class_combinations = list(itertools.combinations(classlist, 2))

            for i, (c1, c2) in enumerate(class_combinations):
                #índices de las muestras con clase c1 y clase c2
                index_c1 = np.where(y == c1)
                index_c2 = np.where(y == c2)
                
                #trials correspondientes a las clases c1 y c2
                trials_c1 = X[index_c1]
                trials_c2 = X[index_c2]
                
                #concatenamos los trials a utilizar para entrenar el filtro CSP
                trials = np.concatenate((trials_c1, trials_c2), axis = 0)
                
                #concatenamos las etiquetas de los trials
                labels = np.concatenate((np.ones(trials_c1.shape[0]), np.zeros(trials_c2.shape[0])), axis = 0)
                
                #fitteamos el filtro CSP
                self.csplist[i].fit(trials, labels)

        if self.method == "ova":
            classlist = np.unique(y)
            for i, c in enumerate(classlist):
                
                c_index = np.where(y == c) #índices de la clase de interés
                others_index = np.where(y != c) #índices de las demás clases
                
                c_trials = X[c_index] #trials de la clase de interes
                others_trials = X[others_index] #trials de las demás clases
                
                c_labels = np.zeros(c_trials.shape[0]) #etiquetas de los trials de la clase de interés
                others_labels = np.ones(others_trials.shape[0]) #etiquetas de los trials de las demás clases
                labels = np.concatenate((c_labels, others_labels), axis = 0) #concatenamos las etiquetas
                
                #fitteamos el filtro CSP
                self.csplist[i].fit(c_trials, labels)

        return self #el método fit siempre debe retornar self
    
    def transform(self, X, y = None):
        """Para cada csp en self.csplist, se transforman los trials de X
        
        - X: ndarray. Es un arreglo de numpy con los datos de entrenamiento. Tiene el formato (n_trials, n_channels, n_samples)"""

        #transformamos los trials de cada clase
        X_transformed = [csp.transform(X) for csp in self.csplist]
        
        #concatenamos los trials de cada clase
        X_transformed = np.concatenate(X_transformed, axis = 0)
        
        return X_transformed


if __name__ == "__main__":

    sample_frec = 100.
    c3, cz, c4 = 26, 28, 30 #canales de interés

    folder = "testData/"
    left = np.load(folder+"all_left_trials.npy", allow_pickle=True)
    right = np.load(folder+"all_right_trials.npy", allow_pickle=True)
    print(left.shape) #[n_channels, n_samples, ntrials]
    print(right.shape) #[n_channels, n_samples, ntrials]

    #los datos deben pasarse al csp de la forma [n_epochs, n_channels, n_samples]
    #los datos que cargamos en eegmatrix están de la forma [n_clases, n_channels, n_samples]

    #Contactemos los trials de cada clase en un sólo array
    eegmatrix = np.concatenate((left,right), axis=0) #importante el orden con el que concatenamos
    print(eegmatrix.shape) #[ n_trials (o n_epochs), n_channels, n_samples]

    class_info = {1: "left", 2: "right"} #diccionario para identificar clases. El orden se corresponde con lo que hay eneegmatrix
    n_clases = len(list(class_info.keys()))

    #genero las labels
    n_trials = left.shape[0]
    totalTrials = eegmatrix.shape[0]
    labels = np.array([np.full(n_trials, label) for label in class_info.keys()]).reshape(totalTrials)
    print(labels.shape)
    print(labels) #las labels se DEBEN corresponder con el orden de los trials en eegmatrix

    ## **************************************************
    #Separamos los datos en train, test y validación
    from sklearn.model_selection import train_test_split

    eeg_train, eeg_test, labels_train, labels_test = train_test_split(eegmatrix, labels, test_size=0.2, random_state=1)
    eeg_train, eeg_val, labels_train, labels_val = train_test_split(eeg_train, labels_train, test_size=0.2, random_state=1)
    
    ## **************************************************

    #instanciamos el objeto CSPMulticlass
    # cspmulticlass = CSPMulticlass(n_components = 3, method = "ovo", n_classes = len(np.unique(labels)),
    #                               transform_into="csp_space", reg = 0.03, component_order="mutual_info")

    cspmulticlass = CSPMulticlass(n_components=2, method = "ovo", n_classes = len(np.unique(labels)), reg=None, log=None, norm_trace=False)
    print(f"Cantidad de filtros CSP a entrenar: {len(cspmulticlass.csplist)}")

    #entrenamos el csp
    cspmulticlass.fit(eeg_train, labels_train)

    eeg_test_transformed = cspmulticlass.transform(eeg_val)
    eeg_test_transformed.shape

    #Extraemos las envolventes de los trials antes y después de aplicar CSP sobre el set de testeo
    from FeatureExtractor import FeatureExtractor

    leftFE = FeatureExtractor(method = "hilbert", sample_rate=100., axisToCompute=2)
    rightFE = FeatureExtractor(method = "hilbert", sample_rate=100., axisToCompute=2)

    #Features separadas por etiqueta
    left_test_envelope_withCSP = leftFE.fit_transform(eeg_val[labels_val == 1])
    right_test_envelope_withCSP = rightFE.fit_transform(eeg_val[labels_val == 2])

    #Graficamos las features para las componentes 1 y 2 obtenidas por el CSP
    import matplotlib.pyplot as plt

    sample_frec = 100.
    t1 = -0.5
    t2 = 2.5
    timeline = np.arange(t1,t2,1/sample_frec)

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle("Envolventes set de validación con CSP - Promedio sobre trials")
    ax[0].plot(timeline, left_test_envelope_withCSP.mean(axis = 0)[0], label = "1er componente (izq)")
    ax[0].plot(timeline, right_test_envelope_withCSP.mean(axis = 0)[0], label = "1er componente (der)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(timeline, left_test_envelope_withCSP.mean(axis = 0)[1], label = "2da componente (izq)")
    ax[1].plot(timeline, right_test_envelope_withCSP.mean(axis = 0)[1], label = "2da componente (der)")
    ax[1].legend()
    ax[1].grid()
    plt.plot()
    plt.show()

    #Extraemos las potencias de las componentes poryectadas luego de aplicar CSP al set de testeo
    leftFE = FeatureExtractor(method = "welch", sample_rate=100., axisToCompute=2)
    rightFE = FeatureExtractor(method = "welch", sample_rate=100., axisToCompute=2)

    left_test_powerCSP = leftFE.fit_transform(eeg_val[labels_val == 1])
    right_test_powerCSP = rightFE.fit_transform(eeg_val[labels_val == 2])

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle("Potencia set de validación con CSP - Promedio sobre trials")
    ax[0].plot(left_test_powerCSP.mean(axis = 0)[0], label = "1er componente (izq)")
    ax[0].plot(right_test_powerCSP.mean(axis = 0)[0], label = "1er componente (der)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(left_test_powerCSP.mean(axis = 0)[1], label = "2da componente (izq)")
    ax[1].plot(right_test_powerCSP.mean(axis = 0)[1], label = "2da componente (der)")
    ax[1].legend()
    ax[1].grid()
    plt.plot()
    plt.show()