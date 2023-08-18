import numpy as np
import logging
import pandas as pd
import TrialsHandler

# Implementing the modified class now
class ModifiedTrialsHandlerWithSuper(TrialsHandler):
    def __init__(self, *args, **kwargs):
        super(ModifiedTrialsHandlerWithSuper, self).__init__(*args, **kwargs)
    
    def getTrials(self):
        """Función modificada para extraer los trials dentro de self.rawEEG, tomando 10 segundos para cada trial."""
        
        # Calculamos la cantidad de muestras que representa el tinit y tmax
        tinit_samples = int(self.tinit * self.sample_rate)
        tmax_samples = int(self.tmax * self.sample_rate)

        # Calculamos la cantidad de trials
        trials = self.eventos.shape[0]
        # Calculamos la cantidad de canales
        channels = self.rawEEG.shape[0]
        # Calculamos la cantidad total de muestras por trial
        total_samples = tmax_samples

        # Creamos un array de numpy para almacenar los trials
        trialsArray = np.zeros((trials, channels, total_samples))

        # Recorremos los trials
        for trial in self.eventos.index:
            # Calculamos la cantidad de muestras que representa el startingTime.
            startingTime_samples = int(self.eventos.loc[trial]["startingTime"] * self.sample_rate)
            # Usamos startingTime_samples como punto de inicio para extraer las muestras
            trialsArray[trial-1] = self.rawEEG[:, startingTime_samples : startingTime_samples + total_samples]

        print("Se han extraido {} trials".format(trials))
        print("Se han extraido {} canales".format(channels))
        print("Se han extraido {} muestras por trial".format(total_samples))

        return trialsArray
    