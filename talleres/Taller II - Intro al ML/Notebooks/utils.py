from json import load
import os
import pandas as pd
import numpy as np

def loadDS(path, filename):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinarCaracteristicas(BaseEstimator, TransformerMixin):
    def __init__(self, agregarBaniosPorHabitacion = True):
        self.agregarBaniosPorHabitacion = agregarBaniosPorHabitacion
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        habutacionPorHogar = X[:, rooms_ix] / X[:, households_ix]
        habitantesPorDistrito = X[:, population_ix] / X[:, households_ix]
        if self.agregarBaniosPorHabitacion:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, habutacionPorHogar, habitantesPorDistrito, bedrooms_per_room]
        else:
            return np.c_[X, habutacionPorHogar, habitantesPorDistrito]
