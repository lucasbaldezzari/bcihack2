# Scripts Python - Revisión 8/3/2023

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

Esta gestión se hace con la clase *[EEGLogger.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/EEGLogger/eegLogger.py)*.

#### Responsable

- [x] Lucas Baldezzari 

## Bloque de procesamiento de señal - Signal Processor Block

Se encarga de,

1) Filtrar la señal proveniente del bloque *EEGLogger* con el módulo *[Filter.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/Filter.py)*. Filtrado de la señal con un pasabanda y un notch.
2) Extraer las características con el módulo *[FeatureExtractor.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/FeatureExtractor.py)*. Aquí se ejecutarán los algoritmos necesarios para selección de canal (aplicando CSP, ICA, o diferentes técnicas aún a definir).
3) Clasificar la señal con *Classifier.py*. Esta clase lo que hará es cargar un clasificador ya entrenado para clasificar lo preveniente del módulo de extracción de características $^1$. 

Al momento se implementa una clase para intentar mejorar la extracción de características a través de Common Spatial Pattern. La clase es *[CommonSpatialPatter](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/CSP.py)*. La misma hace uso de la clase [CSP](https://mne.tools/stable/generated/mne.decoding.CSP.html) de MNE y posee métodos _fit()_, _transform()_ y _fit\_transform()_.

**NOTA:** Las clases dentro del bloque de _SignalProcessor_ se implementan como si fueran _[Transformers](https://scikit-learn.org/stable/data_transforms.html)_ de ScikitLearn (heredan de BaseEstimator, TransformerMixin). La idea es poder utilizar estos objetos en un _[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)_, lo que nos da la ventaja de probar diferentes estrategias de manera rápida y sencilla.

$^1$_Importante:_ Queda pendiente definir si se usará esta clase o directametne un clasificador de sklearn.

#### Responsable

- [x] Lucas Baldezzari 

## Bloque para transmitir/recibir mensajes- Messenger Block

Bloque para comunicación entre PC y el ejecutor (que será un brazo robótico y una silla de ruedas). Los comandos obtenidos por el bloque de clasificación son enviados al dispositivo a controlar. El bloque de mensajería tambien puede recibir mensajes desde el dispositivo controlado.

#### Responsable

- [x] Lucas Baldezzari 

## GUI Block

### Responsables

- [x] Emiliano Álvarez
- [x] Lucas Baldezzari

### Dependencias

La versión de python a utilizar será *>= 3.10.1*

- Dependencias necesarias para ejecutar, probar y ejecutar estos scripts (*Lucas*)...

**TODO**