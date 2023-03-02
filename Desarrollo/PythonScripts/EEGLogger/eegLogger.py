import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import logging, argparse
import time


class EEGLogger():
    """Clase para adquirir/registrar señales de EEG a partir de las placas Cyton o Ganglion de OpenBCI"""

    def __init__(self, board, board_id) -> None:
        """Constructor de la clase
        - board: objeto de la clase BoardShim
        - board_id: id de la placa. Puede ser BoardIds.CYTON_BOARD.value o BoardIds.GANGLION_BOARD.value o BoardIds.SYNTHETIC_BOARD.value
        """

        self.board = board
        self.board_id = board_id
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id) #frecuencia de muestreo
        self.acel_channels = BoardShim.get_accel_channels(board_id) #canales de acelerometro
        self.gyro_channels = BoardShim.get_gyro_channels(board_id) #canales de giroscopio
        self.rawData = np.zeros((len(self.eeg_channels), 0)) #datos crudos
        pass
        
    def connectBoard(self):
        """Nos conectamos a la placa y empezamos a transmitir datos"""
        self.board.prepare_session()
        print("***********")
        print(f"Channles: {self.eeg_channels}")
        print("***********")
        self.board.start_stream()
        pass

    def stopBoard(self):
        """Paramos la transmisión de datos de la placa"""
        self.board.stop_stream()
        self.board.release_session()
    
    def getSamples(self, sampleLength = 6):
        """Obtenemos algunas muestras de la placa. La cantidad de muestras que devuelve el método depende del timeLength y de la frecuencia
        de muestro de la placa. 
        Los datos se entregan en un numpy array de forma [canales, muestras]. Los datos están en microvolts.
        - sampleLength: duración (en segundos) de la señal a adquirir de la placa. Por defecto son 6 segundos."""

        num_samples = self.board.get_sampling_rate(self.board_id) * sampleLength
        return self.board.get_board_data(num_samples)
    
    def addData(self, newdata):
        """Agregamos datos a la variable rawData. 
        - newdata: numpy array de forma [canales, muestras]"""
        self.rawData = np.concatenate((self.rawData, newdata), axis = 1)
    
    def saveData(self, fileName = "subject1.npy", path = "recordedEEG/"):
        """Guardamos los datos crudos en un archivo .npy
        - fileName: nombre del archivo
        - path: carpeta donde se guardará el archivo"""

        with open(path + fileName, "wb") as f:
            np.save(f, self.rawData)


def main():

    placas = {"cyton": BoardIds.CYTON_BOARD, #IMPORTANTE: frecuencia muestreo 256Hz
              "ganglion": BoardIds.GANGLION_BOARD, #IMPORTANTE: frecuencia muestro 200Hz
              "synthetic": BoardIds.SYNTHETIC_BOARD}
    
    placa = placas["synthetic"]

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    #IMPORTENTE: Chequear en que puerto esta conectada la OpenBCI.  
    puerto = "COM5"
    
    parser = argparse.ArgumentParser()
    
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')

    
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default = puerto)
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default = placa)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
        
    board = BoardShim(args.board_id, params) #genero un objeto para control de placas de Brainflow

    eeglogger = EEGLogger(board, args.board_id) #instanciamos un objeto para adquirir señales de EEG desde la placa OpenBCI
    eeglogger.connectBoard() #nos conectamos a la placa y empezamos a transmitir datos

    trialDuration = 2 #duración del trial en segundos

    print("Adquiriendo datos por primera vez...")
    print("Debemos esperar para completar el buffer")
    time.sleep(trialDuration) #esperamos a que se adquieran los datos

    newData = eeglogger.getSamples(trialDuration)
    # print(newData)
    print("Forma del array de datos [canales, muestras]: ",newData.shape)
    eeglogger.addData(newData[:16])

    print("Guardando datos...")
    eeglogger.saveData(fileName = "subject1.npy", path = "recordedEEG/") #guardamos los datos en un archivo .npy

    print("Detener la adquisición de datos")
    eeglogger.stopBoard()
    

if __name__ == "__main__":
    main()