# Scripts Python

En esta sección se encuentra el código fuente de los diferentes scripts en python para implemetar la ICC.

La siguiente imágen muestra el diagrama de bloques V1.0 de la BCI.

![Diagrama de bloques](bloques.png)

A continuación se resume lo contenido dentro de cada directorio.

La versión de python a utilizar será *>= 3.10.1*

## CoreModule

Este módulo es el gestor y administrador de los principales procesos de la BCI, entre los que podemos destacar la escritura/lectura de datos -EEG, clasificadores-, gestión/escritura/lectura de eventos relevantes, procesamiento de EEG (online), gestión de las diferentes etapas de un trial.
Se comunica con todos los módulos.

Este módulo es el encargado de gestionar los siguientes procesos,

- Gestión de los trials. Temporización de cada etapa de un trial -tiempo para mostrar qué tarea debe ejecutar la persona, tiempo de estimulación, tiempo de descanso-.	


### Responsable

- Lucas Baldezzari 

### Dependencias

- Dependencias necesarias para ejecutar, probar y correr la GUI-> (*Lucas*)

## GUIModule

### Responsables
- Emiliano Álvarez
- Lucas Baldezzari

**TODO**

- Resumen -> (*Lucas*)
- Dependencias necesarias para ejecutar, probar y correr la GUI-> (*Emi*) a la fecha 24/02/2023:
- python==3.9.7
- pip
- pip:
  - PyQT5==5.15.6
  - pyqtgraph==0.13.1
  - numpy==1.20.3
  - brainflow==5.0.0
  - scipy==1.7.1

## SignalModule 

### Responsable

- Lucas Baldezzari 

**TODO**

- Resumen -> (*Lucas*)
- Dependencias necesarias para ejecutar, probar y correr la GUI-> (*Lucas*)
