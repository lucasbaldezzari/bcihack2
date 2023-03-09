"""IMPORTANTE: Esta clase aún esta en revisión"""
import threading
import time
import numpy as np
import scipy.io as sio
import pickle

class Classifier(threading.Thread):
    """Clase para clasificar los datos. Esta clase usará los datos extraídos por el bloque FeatureExtractor para clasificarlos."""
    
    def __init__(self, actionList, filename, modelPath = "modelos/", sleepTime = 0.01):
        """Constructor de la clase
        
        Atributos:
        - actionList (list): Lista con los datos a ser clasificados. Estos datos se corresponden con los movimientos ejecutados/imaginados
            por la persona. El clasificador nos arrojará un valor que se corresponderá con el movimiento que se está ejecutando/imaginando. Este valor se
            utilizará para seleccionar una acción a partir dentro de actionList.
        - filename (str): Nombre del archivo donde se encuentra el modelo a utilizar. Formato pickle.
        - classificationResult (list): Lista de los resultados de la clasificación.
        - model: Modelo a utilizar por el clasificador
        - modelPath (str): Ruta a donde buscar el modelo
        - NewFeatures (bool): Indica si hay nuevas features para clasificar."""

        threading.Thread.__init__(self)
        self.actionList = actionList
        self.classificationResults = [] #cada clasificación se agrega a esta lista
        self.model = self.loadModel(filename) #cargamos el modelo a utilizar
        self.NewFeatures = False
        self.sleepTime = sleepTime

    def run(self):
        """Iniciamos el hilo de la clase"""
        
        while True:
            if self.NewData:
                self.classification = self.model.predict(self.data)
                self.classificationResult = self.model.decision_function(self.data)
                self.data = []
                self.NewFeatures= False
            time.sleep(self.sleepTime)

    def loadModel(self, filename):
        """Carga el modelo a utilizar por el clasificador."""
        with open(self.modelName, 'rb') as file:
            self.model = pickle.load(file)

    def getClassification(self, features):
        """Retorna la clasificación a partir de las features.
        -features: Es un numpy array de la forma [n_channels, n_samples]"""

        self.classificationResults.append(self.model.predict(features)) #clasificamos

        return self.classificationResults[-1] #retornamos el último valor de la lista

def main():

    with open("testing_features.npy", "rb") as f:
        features = np.load(f)

    clasificador = Classifier()
    clasificador.start()

    #clasificamos para obtener una acción
    action = clasificador.getClassification(features)

    print(action)

if __name__ == '__main__':
    main()

