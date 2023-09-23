from keras.models import Sequential
from keras.layers import (Conv2D, BatchNormalization, ELU, AveragePooling2D,
                          Dropout, Permute, Reshape, GRU, Dense, DepthwiseConv2D,
                          Activation, SeparableConv2D)
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.regularizers import l2
from keras.optimizers import Adam

class CRNN_EEGNet:
    
    def __init__(self, input_shape, num_classes, F1=8, D=2, dropout_rate=0.4, l2_lambda=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.F1 = F1
        self.D = D
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.model = self._build_model()

    def _build_model(self):
        C, T = self.input_shape
        model = Sequential()

        # Bloque 1: Capas convolucionales
        model.add(Reshape((C, T, 1), input_shape=self.input_shape))
        model.add(Conv2D(self.F1, (C, 1), padding='same', activation='linear', kernel_regularizer=l2(self.l2_lambda)))
        model.add(BatchNormalization())
        model.add(DepthwiseConv2D((1, T), depth_multiplier=self.D, depthwise_constraint='max_norm', activation='linear', depthwise_regularizer=l2(self.l2_lambda)))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(AveragePooling2D((4, 1)))
        model.add(Dropout(self.dropout_rate))

        # Capas recurrentes
        model.add(Permute((1, 3, 2)))
        model.add(Reshape((-1, self.F1 * self.D)))
        model.add(GRU(64, dropout=self.dropout_rate))

        # Capa clasificadora
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(self.l2_lambda)))

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