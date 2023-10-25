from keras.models import Sequential
from keras.layers import (Conv2D, BatchNormalization, ELU, AveragePooling2D,
                          Dropout, Permute, Reshape, GRU, Dense, DepthwiseConv2D,
                          Activation, SeparableConv2D,MaxPooling2D,Flatten)
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.regularizers import l2
from keras.optimizers import Adam
import keras

class CRNN_EEGNet:
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        # Primera capa convolucional
        # Primera capa convolucional con padding
        # Primera capa convolucional con padding
        model = Sequential()

# Primera capa convolucional con padding
              # Primera capa convolucional con padding
        model.add(Conv2D(32, kernel_size=(1, 3), activation='relu', padding='same', input_shape=(6, 750, 1)))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.25))

        # Segunda capa convolucional con padding
        model.add(Conv2D(64, kernel_size=(1, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.25))

        # Aplanar y agregar capas densas
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        # Compilar el modelo
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        return model

    def compile(self, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', metrics=['accuracy']):
        """
        Método para compilar el modelo.
        ...
        """
        if optimizer == 'adam':
          optimizer_instance = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, batch_size=32, epochs=10, validation_data=None,
        patience=10, model_checkpoint_path="best_model.h5"):
        """
        Método para entrenar el modelo CRNN_EEGNet.

        Parámetros:
            - x_train (array): Datos de entrenamiento.
            - y_train (array): Etiquetas de entrenamiento.
            - batch_size (int, opcional): Tamaño del lote para el entrenamiento. Por defecto es 32.
            - epochs (int, opcional): Número máximo de épocas para entrenar el modelo. Por defecto es 10.
            - validation_data (tuple, opcional): Datos de validación en el formato (x_val, y_val).
                                                Si se proporciona, se usa para validar el modelo después de cada época.
            - patience (int, opcional): Número de épocas sin mejora en la pérdida de validación para activar
                                        el early stopping. Por defecto es 10.
            - model_checkpoint_path (str, opcional): Ruta donde se guardará el mejor modelo basado en la pérdida
                                                    de validación. Por defecto es "best_model.h5".

        Devuelve:
            - history (History): Objeto con los registros del entrenamiento, que incluye la pérdida y las métricas
                                para cada época.

        Nota:
            Este método utiliza early stopping para prevenir el sobreajuste. Si la pérdida de validación no mejora
            durante un número de épocas especificado en 'patience', se detendrá el entrenamiento y se restaurarán
            los pesos del modelo al mejor estado encontrado. Además, se guardará el mejor modelo durante el
            entrenamiento en el archivo especificado en 'model_checkpoint_path'.
        """
      # Verificar que se proporciona validation_data
        if validation_data is None:
            raise ValueError("validation_data must be provided when monitoring val_loss")

        # Definir callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

        # Entrenar el modelo
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[early_stopping, model_checkpoint]
        )
        return self.history

    def evaluate(self, x_test, y_test):
        """
        Método para evaluar el modelo en datos de prueba.
        ...
        """
        return self.model.evaluate(x_test, y_test)

    def summary(self):
        """
        Método para imprimir un resumen de la arquitectura del modelo.
        """
        self.model.summary()