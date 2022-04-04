# Proyecto de Visión Artificial

## Proyecto

Se han programado una serie de scripts en python que resuelven problemas de visión artificial haciendo uso de la librería de OpenCV.

## Diseño

Cada script diseñado está comentado en un archivo notebook de Jupiter. En estos archivos se puede encontrar información de diseño e implementación, así como pruebas ejemplos y resultados de la aplicación de los mismos.

### Notebooks

| Ejercicio | Descripción |
|---:|---|
| Calibración | Se hace uso de imágnes de tamaño conocido para realizar una calibración de imágen |
| Actividad | Detecta y evalua zonas de actividad de imágenes |
| Color | Detecta objetos usando curvas ROI |
| Filtros | Aplicación de diferentes filtros a imágenes |
| Rectif | Rectifica la medición de distancias mediante el uso de análisis de imágenes y distancias |
| Pano | Ejecuta un análisis de perspectivas de imágenes para elaborar una panorámica de imágenes homográficas  |
| RA | Genera un objeto 3D en una superficie con puntos de referencia conocidos |

### Entorno e Instalación

Este proyecto se ha elaborado con el programa PyCharm del paquete InteliJ. La configuración se realiza mediante la instalación del archivo de requisitos _requierements.txt_. Se instalan las librerias necesarias para la ejecución del programa.

> **Requisitos**: jupyter notebook, numpy, opencv-python

```shell
# python<version> -m venv <virtual_env_name>
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Fuentes de Información

[VIA UMU](https://dis.um.es/profesores/alberto/vision.html)

[Repositorio VIA](https://github.com/albertoruiz/umucv)
