# Repositorio  Hackathon BCI 2022/2023

#### Credits

```diff

- Please, if you will use scripts, information or anything from this repository, please give us some credits.
+ We will appreciate it!

- https://www.linkedin.com/in/lucasbaldezzari/

+ Thanks!
```

## Resumen proyecto
Haackathon de BCI para comandar dispositivos utilizando Imaginería Motora.

El siguiente repositorio será utilizado para almacenar bibliografía, set de datos, firmware, hardware, imágenes, entre otros recursos correspondiente al hackathon de BCI 2022/2023 de la UTEC, Uruguay.

**NOTA**: Queda pendiente mejorar el resumen del proyecto.

## Director y autor del proyecto

[MSc. Bioing. BALDEZZARI Lucas](https://www.linkedin.com/in/lucasbaldezzari/)

## Colaboradores

- [Dra. SALUM Graciela - graciela.salum@utec.edu.uy](https://www.linkedin.com/in/graciela-marisa-salum-5262bb47)
- [Téc. SUAREZ Tomy - tomy.suarez@utec.edu.uy](https://www.linkedin.com/in/tomy-suarez-a06993162)
- [Téc. ÁLVAREZ Emiliano - emiliano.alvarez@utec.edu.uy](https://www.linkedin.com/in/emilianoalvarezruiz)
- [Téc. MAR Walter - walter.mar@utec.edu.uy](https://www.linkedin.com/in/walter-mar-6b2104195/?originalSubdomain=uy)

### Demo sistema

Demostración del sistema al día 23/5/2023.

![animation](https://github.com/lucasbaldezzari/bcihack2/assets/21134083/7acf2000-d3af-4217-a934-7afd6348635d)

## Discord

Puedes visitarnos en Discord en nuestro canal [Uruguay BCI](https://discord.gg/7e6ZdFgh).

# [Scripts Python - Revisión 30/4/2023](https://github.com/lucasbaldezzari/bcihack2/tree/main/Desarrollo/PythonScripts)

En esta sección se encuentra el código fuente de los diferentes scripts en python para implementar la ICC.

La siguiente imagen muestra el diagrama de bloques V1.4 de la BCI.

![Diagrama de bloques](/Desarrollo/PythonScripts/figures/bloques.png)

A continuación, se resume lo contenido dentro de cada directorio.

## Bloque principal - Core Block

Este módulo es el gestor y administrador de los principales procesos de la BCI, entre los que podemos destacar la escritura/lectura de datos -EEG, clasificadores-, gestión/escritura/lectura de eventos relevantes, procesamiento de EEG (online), gestión de las diferentes etapas de un trial.
Se comunica con todos los módulos.

Este módulo es el encargado de gestionar los siguientes procesos,

- Según el modo de trabajo seleccionado (entrenamiento, calibración, online) se configuran los diferentes parámetros de la sesión (duración trials, duración de tiempo de descanso, nombre de sesión, número de sujeto, entre otros).
- Gestión de los trials en cada sesión. Temporización de cada etapa de un trial -tiempo para mostrar qué tarea debe ejecutar la persona, tiempo de estimulación, tiempo de descanso-.
- Envía y recibe *eventos* hacia y desde los diferentes bloques. Estos eventos se registran para un posterioir análisis, pero también para el control de los bloques.

#### Responsable

- [x] MSc. Bioing. Lucas Baldezzari

## Bloque de adquisición de señal - EEGLogger Block

Bloque para comunicación con placas OpenBCI a través de Brainflow. Este bloque permite la adquisición y registro de señales de EEG (rawEEG), información del acelerómetro, entre otros. Se comunica con el bloque de procesamiento de señal.

Esta gestión se hace con la clase *[EEGLogger.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/EEGLogger/eegLogger.py)*.

#### Responsable

- [x] MSc. Bioing. Lucas Baldezzari 

## Bloque de procesamiento de señal - Signal Processor Block

A continuación de mencionan las funcionalidades y clases del bloque,

### Primer filtrado de señal

Se aplican filtros pasabanda y notch a señal proveniente del bloque *EEGLogger* con la clase *[Filter.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/Filter.py)*.

### Filtrado espacial - CSPMulticlass

Esta clase aplica un filtrado espacial a través de [Common Spatial Pattern](https://en.wikipedia.org/wiki/Common_spatial_pattern#:~:text=Common%20spatial%20pattern%20(CSP)%20is,in%20variance%20between%20two%20windows). La misma hace uso de la clase [CSP](https://mne.tools/stable/generated/mne.decoding.CSP.html), pero se le agregan algunos métodos adicionales. La clase engargada es [CSPMulticlass](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/CSPMulticlass.py).

La clase recibe los datos filtrados por [Filter.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/Filter.py) y genera una lista de filtros CSP. La cantidad de filtros espaciales creados depende, en primera instancia de la cantidad de clases que se tenga, y también del tipo de comparación que se quiere hacer. 

Se pueden utilizar dos enfoques para generar y aplicar los filtros espaciales. Estos son,

- One vs One: La clase genera $\frac{K(K-1)}{2}$ filtros diferentes ($K$ es la cantidad de clases) cuando se llama al método _fit()_.
- One vs All: La clase genera $K$ filtros diferentes ($K$ es la cantidad de clases) cuando se llama al método _fit()_.

Cuando se llama a *_transform(signal)_* el método aplica cada filtro dentro _csplist_ a los datos dentro de _signal_ y retorna un array con estas proyecciones concatenadas de tal forma que el array retornado tiene la forma [n_trials, n_components x n_filters, n_samples]. 

A modo de ejemplo, si tenemos 4 clases, y entrenamos un _CSPMulticlass_ para dos componentes y con método _ovo_, deberíamos tener $2_{componentes}\times6_{filtros} = 12$ nuevas dimensiones, entonces el array que entregará CSPMulticlass al aplicar _transform_ es [n_trials, 12, n_samples]. 

### Extracción de características

La extracción de características está a cargo del módulo *[FeatureExtractor.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/FeatureExtractor.py)*.

La extracción de caraterísticas puede hacerse por dos métodos, uno es obteniendo la densidad espectral de potencia por método de [Welch](https://en.wikipedia.org/wiki/Welch%27s_method) y la otra es la envolvente de la señal de EEG a través de la transformada de [Hilbert](http://www.scholarpedia.org/article/Hilbert_transform_for_brain_waves).

Se plantean dos enfoques para la aplicación del CSP y la extracción de características. Una es la estrategia *OneVsOne* y la otra es *OneVsRest*.

La siguiente figura (adaptada de [Multiclass Classification Based on Combined Motor Imageries](https://www.frontiersin.org/articles/10.3389/fnins.2020.559858/full)) muestra un diagrama de aplicación de CSP y extracción de características para la fase de entrenamiento, las fases de feedback y online son similares, sólo que los CSP no se entrenan, sino que se utilizan filtros previamente entrenados durante la fase de entrenamiento.

Se utilizan las señales de EEG previamente filtradas (pasabanda y notch), trials y labels para obtener los filtros espaciales que proyectarán el EEG a un nuevo espacio. La cantidad de filtros espaciales a obtener está en función del número de clases según la cantidad de clases y el método de comparación seleccionado.

A partir de las salidas de estos filtros se extraen sus características con *[FeatureExtractor.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/FeatureExtractor.py)* y se concatenan cada una de estas para formar el **feature vector** final.

![Diagrama aplicación de CSP y Extracción de características - OvO](/Desarrollo/PythonScripts/figures/cspovotrain.png)

El entrenamiento y aplicación de filtrado por CSP está a cargo de [CSPMulticlass.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/CSPMulticlass.py).

La concatenación de las features en un único feature se hace con la clase [RavelTransformer](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/RavelTransformer.py).

#### Graficando Patrones y Filtros

La clase [CSPMulticlass](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/CSPMulticlass.py) posee métodos para graficar los mapas topográficos referentes a los filtros y patrones obtenidos a partir de entrenar la clase con _fit()_.

La cantidad de patrones o filtros a graficar depende de la cantidad de clases, de la cantidad de componentes y de si los CSP se obtienen a partir de entrenar las clases _one vs one_ o _one vs all_.

Las siguientes figuras muestran ejemplos de patrones y filtros para el caso de entrenar el *CSPMulticlass* para 5 clases y tres componentes por clase.


*Patrones*
<img src="/Desarrollo/PythonScripts/figures/patterns.png" width="680" alt = "Patrones CSP" class="center"/><img>



*Filtros*
<img src="/Desarrollo/PythonScripts/figures/filters.png" width="680" alt = "Filtros CSP" class="center"/>><img>


### Clasificación
Se entrenan y utilizan clasificadores de la librería Scipy.

Al momento se implementa una clase para intentar mejorar la extracción de características a través de Common Spatial Pattern. La clase es *[CommonSpatialPatter](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/SignalProcessor/CSP.py)*. 

**NOTA:** Las clases dentro del bloque _SignalProcessor_ se implementan como si fueran _[Transformers](https://scikit-learn.org/stable/data_transforms.html)_ de ScikitLearn (heredan de BaseEstimator, TransformerMixin). La idea es poder utilizar estos objetos en un _[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)_, lo que nos da la ventaja de probar diferentes estrategias de manera rápida y sencilla.

En el [classifierPipeline.py](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/classifierPipeline.py) se muestra una aplicación completa para el análisis de las señales de EEG registradas sobre uno de nuestros voluntarios.

Se muestra un resumen debajo,

```python
### ********** Creamos el pipeline para LDA **********

pipeline_lda = Pipeline([
    ('pasabanda', filter),
    ('cspmulticlase', cspmulticlass),
    ('featureExtractor', featureExtractor),
    ('ravelTransformer', ravelTransformer),
    ('lda', lda)
])

### ********** Grilla de ejemplo **********
param_grid_lda = {
    'pasabanda__lowcut': [5, 8],
    'pasabanda__highcut': [12],
    'cspmulticlase__n_components': [2],
    'cspmulticlase__method': ["ovo","ova"],
    'cspmulticlase__n_classes': [len(np.unique(labels))],
    'cspmulticlase__reg': [0.01],
    'cspmulticlase__log': [None],
    'cspmulticlase__norm_trace': [False],
    'featureExtractor__method': ["welch", "hilbert"],
    'featureExtractor__sample_rate': [fm],
    'featureExtractor__band_values': [[8,12]],
    'lda__solver': ['svd'],
    'lda__shrinkage': [None],
    'lda__priors': [None],
    'lda__n_components': [None],
    'lda__store_covariance': [False],
    'lda__tol': [0.0001, 0.001],
}

#Creamos el GridSearch para el LDA
grid_lda = GridSearchCV(pipeline_lda, param_grid_lda, cv=5, n_jobs=1, verbose=1)

### ********** Entrenamos el modelo **********
grid_lda.fit(eeg_train, labels_train)
### ******************************************
```

#### Responsable

- [x] MSc. Bioing. Lucas Baldezzari 

## Bloque para transmitir/recibir mensajes- Messenger Block

Bloque para comunicación entre PC y el ejecutor (que será un brazo robótico y una silla de ruedas). Los comandos obtenidos por el bloque de clasificación son enviados al dispositivo a controlar. El bloque de mensajería tambien puede recibir mensajes desde el dispositivo controlado.

#### Responsable

- [x] MSc. Bioing. Lucas Baldezzari 

## GUI Block

Existen 3 GUIs o APPs

- [ConfigAPP](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/GUIModule/ConfigAPP.py): Esta interfaz permite la configuración de diferentes parámetros de la sesión, por ejemplo, número sesión, tipo de sesión (ejecutar o imaginar), placa de adquisición a usar, canales a usar, parámetros de filtrado, pipeline de clasificación, entre varios más.
- [IndicatorAPP](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/GUIModule/IndicatorAPP.py): Esta clase permite mostrar en pantalla qué tarea o _cue_ debe ejecutar la persona. Además, para las sesiones de calibración/feedback se tiene una _barra_ que muestra la probabilidad, de 0 a 100%, de que la clase clasificada se corresponda con la que se solicitó al sujeto que ejecute/imagine.
- [SupervisionAPP](https://github.com/lucasbaldezzari/bcihack2/blob/main/Desarrollo/PythonScripts/scripts/GUIModule/SupervisionAPP.py): Esta clase es usada por el grupo de técnicos/as e ingenieros/as para supervisar diferentes parámetros de la sesión y la señal de EEG que se registra.

#### Responsables

- [x] Tec. Emiliano Álvarez
- [x] MSc. Bioing. Lucas Baldezzari

### Dependencias

La versión de python a utilizar será *>= 3.10.1*

- Dependencias necesarias para ejecutar, probar y ejecutar estos scripts (Completar por *Lucas*).
