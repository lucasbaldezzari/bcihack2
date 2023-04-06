from EEGLogger.eegLogger import eegLogger

from SignalProcessor.Filter import Filter
from SignalProcessor.FeatureExtractor import FeatureExtractor
from SignalProcessor.Classifier import Classifier

class Core():
    """Esta clase es la clase principal del sistema.
    Clase para manejar el bloque de procesamiento de los datos (filtrado, extracción de característica y clasificación),
    las GUI y el bloque de comunicación con el dispositivo de control. 
    Esta clase usará las clases EEGLogger, Filter, FeatureExtractor y Classifier para procesar la señal de EEG.
    NOTA: Queda pendiente ver la implementación de al menos un hilo para controlar el Filter, CSPMultcilass, FeatureExtractor,
    RavelTransformer y Classifier. Este hilo deberá ser el que controle los tiempos de de inicio/inter-trial, el tiempo
    del cue y el de descanso. Además, deberá ser el encargado de enviar los comandos al dispositivo de control (o bien,
    se puede hacer otro para el control de la comunicación con el dispositivo.)
    NOTA 2: Se debe pensar en un hilo para el control de la GUI.
    """
    def __init__(self):
        """Constructor de la clase
        
        NOTA: Definir qué parámetros se necesitan inicar dentro del constructor."""
        # self.eegLogger = eegLogger()
        # self.filter = Filter()
        # self.featureExtractor = FeatureExtractor()
        # self.classifier = Classifier()
        pass

    def loadParameters(self, parameters):
        """Este método se encargará de leer los parámetros importantes para llevar a cabo el control de la aplicación.

        Se propone que la GUI de "Seteo e incio de sesión" pase los parámetros a esta clase en la forma de un diccionario o json.
        
        - Parameters (dict): Diccionario con los parámetros a ser cargados. Los parámetros son:
            -typeSesion (int): Tipo de sesión. 0: Entrenamiento, 1: Feedback o calibración, 2: Online.
            -startingTimes (lista): Lista con los valores mínimo y máximo a esperar antes de iniciar un nuevo cue o tarea. 
            Estos valores se usan para generar un tiempo aleatorio entre estos valores.
            -cueDuration (float): Duración del cue en segundos.
            -finishDuration (float): Duración del tiempo de finalización en segundos.
            -subjectName (str): Nombre del sujeto.
            -sesionNumber (int): Número de la sesión.
            -filterParameters (dict): Diccionario con los parámetros del filtro. Los parámetros son:
                l-owcut (float): Frecuencia de corte inferior del filtro pasa banda.
                -highcut (float): Frecuencia de corte superior del filtro pasa banda.
                -notch_freq (float): Frecuencia de corte del filtro notch.
                -notch_width (float): Ancho de banda del filtro notch.
                -sample_rate (float): Frecuencia de muestreo de la señal.
                -axisToCompute (int): Eje a lo largo del cual se calculará la transformada.
            -featureExtractorMethod (str): Método de extracción de características. Puede ser welch o hilbert.
            -cspFile (str): Ruta al archivo pickle con los filtros CSP. IMPORTANTE: Se supone que este archivo ya fue generado con la sesión
            de entrenamiento y será usado durante las sesiones de feedback y online.
            -classifierFile (str): Ruta al archivo pickle con el clasificador. IMPORTANTE: Se supone que este archivo ya fue generado con la sesión
            de entrenamiento y será usado durante las sesiones de feedback y online.

            Un trial es la suma de startingTimes + cueDuration + finishDuration
            
            NOTA: Se agregarán (o quitarán parámetros a medida que evolucione la clase)."""

        #Parámetros generales para la sesións
        self.typeSesion = parameters["tipySesion"]
        self.startingTimes = parameters["startingTimes"]
        self.cueDuration = parameters["cueDuration"]
        self.finishDuration = parameters["finishDuration"]
        self.subjectName = parameters["subjectName"]
        self.sesionNumber = parameters["sesionNumber"]

        #parámetros del filtro
        self.filterParameters = parameters["filterParameters"]

        ## Parámetros para el CSP
        self.cspComponents = parameters["cspComponents"]
        self.cspType = parameters["cspType"]


    def startEEGLogger(self):
        """Iniciamos el EEGLogger."""
        # self.eegLogger.start()
        pass

    def startFilter(self):
        """Iniciamos el filtro."""
        # self.filter.start()
        pass

    def startFeatureExtractor(self):
        """Iniciamos el FeatureExtractor."""
        # self.featureExtractor.start()
        pass

    def startClassifier(self):
        """Iniciamos el clasificador."""
        # self.classifier.start()
        pass


    def stop(self):
        """Frenamos los hilos."""
        # self.eegLogger.stop()
        # self.filter.stop()
        # self.featureExtractor.stop()
        # self.classifier.stop()
        pass

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