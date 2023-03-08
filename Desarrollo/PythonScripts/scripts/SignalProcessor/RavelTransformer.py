from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

#This class recieve data in the form [n_trials, n_channels or n_components, n_samples] and return data in the form [n_trials, n_components x n_samples]
#the idea is make a ravel of the data in the second dimension and return a matrix with the shape [n_trials, n_components x n_samples] in order to fit a classifier
class RavelTransformer(BaseEstimator, TransformerMixin):
    """Esta clase recibe datos en la forma [n_trials, n_channels or n_components, n_samples] y retorna datos en la forma [n_trials, n_components x n_samples]"""
    def __init__(self, method = "reshape"):

        self.method = method


    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):

        if self.method == "reshape":
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
        elif self.method == "mean":
            X = np.mean(X, axis = 1)

        return X
