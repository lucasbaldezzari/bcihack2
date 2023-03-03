# Scripts Python - Revisión 1/3/2023

En esta sección se encuentra el código fuente de los diferentes scripts en python para implemetar la ICC.

La siguiente imágen muestra el diagrama de bloques V1.4 de la BCI.

![Diagrama de bloques](bloques.png)

A continuación se resume lo contenido dentro de cada directorio.

## Bloque principal - Core Block

Este módulo es el gestor y administrador de los principales procesos de la BCI, entre los que podemos destacar la escritura/lectura de datos -EEG, clasificadores-, gestión/escritura/lectura de eventos relevantes, procesamiento de EEG (online), gestión de las diferentes etapas de un trial.
Se comunica con todos los módulos.

Este módulo es el encargado de gestionar los siguientes procesos,

- Gestión de los trials. Temporización de cada etapa de un trial -tiempo para mostrar qué tarea debe ejecutar la persona, tiempo de estimulación, tiempo de descanso-.	


#### Responsable

- [x] Lucas Baldezzari

## Bloque de adquisición de señal - EEGLogger Block

Bloque para comunicación con placas OpenBCI a través de Brainflow. Este bloque permite la adquisición y registro de señales de EEG (rawEEG), infirmación del acelerómetro, entre otros. Se comunica con el bloque de procesamiento de señal.

Esta gestión se hace con la clase *[EEGLogger.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/EEGLogger/eegLogger.py)*.

#### Responsable

- [x] Lucas Baldezzari 

## Bloque de procesamiento de señal - Signal Processor Block

Se encarga de,

1) Filtrar la señal proveniente del bloque *EEGLogger* con el módulo *[Filter](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/SignalProcessor/Filter.py)*. Filtrado de la señal con un pasabanda y un notch.
2) Extraer las características con el módulo *FeatureExtractor.py*. Aquí se ejecutarán los algoritmos necesarios para selección de canal (aplicando CSP, ICA, o diferentes técnicas aún a definir).
3) Clasificar la señal con *Classifier.py*. Esta clase lo que hará es cargar un clasificador ya entrenado para clasificar lo preveniente del módulo de extracción de características.

**NOTA:** Los clasificadores a utilizar serán entrenados usando clases específicas para cada clasificador a utilizar, como por ejemplo, *SVMClassifier.py*, *LinearRegressionClassifier.py*, *NeuralNetworkClassifier.py*, entre otros posibles.

#### Responsable

- [x] Lucas Baldezzari 

**TODO**

- Resumen -> (*Lucas*)

## Bloque para transmitir/recibir mensajes- Messenger Block

Bloque para comunicación entre PC y el ejecutor (que será un brazo robótico y una silla de ruedas). Los comandos obtenidos por el bloque de clasificación son enviados al dispositivo a controlar. El bloque de mensajería tambien puede recibir mensajes desde el dispositivo controlado.

#### Responsable

- [x] Lucas Baldezzari 

## GUI Block

### Responsables

- [x] Emiliano Álvarez
- [x] Lucas Baldezzari

**TODO**

- Resumen -> (*Lucas*)


### Dependencias

La versión de python a utilizar será *>= 3.10.1*

- Dependencias necesarias para ejecutar, probar y ejecutar estos scripts (*Lucas*)...

**TODO**