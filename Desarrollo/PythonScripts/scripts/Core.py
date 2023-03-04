from EEGLogger.eegLogger import eegLogger

from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.Classifier import Classifier

class Core():
    """Clase para manejar el procesamiento de los datos y la clasificación. Esta clase es la clase principal del sistema.
    Esta clase usará las clases EEGLogger, Filter, FeatureExtractor y Classifier para procesar la señal de EEG.
    Estas clases se ejecutarán en un hilo separado para evitar bloquear el hilo principal
    """
    def __init__(self):
        """Constructor de la clase"""
        self.eegLogger = eegLogger()
        self.filter = Filter()
        self.featureExtractor = FeatureExtractor()
        self.classifier = Classifier()
        
    def start(self):
        """Iniciamos los hilos.
        NOTA: Queda implementar que cada clase pueda ser manejada por threading"""
        # self.eegLogger.start()
        # self.filter.start()
        # self.featureExtractor.start()
        # self.classifier.start()

    def stop(self):
        """Frenamos los hilos."""
        self.eegLogger.stop()
        self.filter.stop()
        self.featureExtractor.stop()
        self.classifier.stop()

    def loadParameters(self, parameters):
        """Este método se encargará de leer los parámetros importantes para llevar a cabo el control de la aplicación.
        
        - Parameters (dict): Diccionario con los parámetros a ser cargados. Los parámetros son:
            -trialDuration (float): Duración de cada trial en segundos.
            -preparationDuration (float): Duración de la preparación en segundos.
            -restDuration (float): Duración del descanso en segundos.
            
            NOTA: Se puede agregar más parámetros si es necesario."""

        #Parametros importantes
        self.trialDuration = parameters["trialDuration"]
        self.preparationDuration = parameters["preparationDuration"]
        self.restDuration = parameters["restDuration"]

    def run(self):
        """Iniciamos el hilo de la clase"""

        #TODO: Implementar el control de la aplicación

        pass

def main():
    # core = Core()
    # core.start()
    pass

if __name__ == "__main__":
    main()