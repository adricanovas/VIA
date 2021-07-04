#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, sourceArgs
from collections import deque
from umucv.util import putText
import sys

def drawPolygon(image, pts, color=(255, 0, 0), thickness=2):
    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv.polylines(
        image, [pts], isClosed, color, thickness)
    return image


def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))

# Leer el archivo
file = open(sys.argv[1], 'r')
lines = file.readlines()
srcimg = lines[0][:-1]
real = lines[1][:-1].split(' ')
preal = list()
for e in real:
    aux = [int(x) for x in e.split(",")]
    preal.append(aux)
rectificados = lines[2][:-1].split(' ')
prectificados = list()
for e in rectificados:
    aux = [int(x) for x in e.split(",")]
    prectificados.append(aux)
escala = float(lines[3])
file.close()

# Carga la imagen
img = cv.imread('../Images/6_RECTIF/' + srcimg)
imgCopy = np.array(img)

points = deque(maxlen=2)

cv.namedWindow('RECTIF')
cv.namedWindow('Resultado')
cv.setMouseCallback('RECTIF', fun)

puntos = np.array(preal)
puntos = puntos.reshape((-1, 1, 2))
# Rectificamos
imgCopy = np.array(img)
imgCopy = cv.polylines(imgCopy, [puntos], isClosed = True, color=(255, 0, 0), thickness=2)

# Mostramos la imagen rectificada
cv.imshow('RECTIF', imgCopy)

# Obtiene la imagen rectificada a partir de la homografía
imgreal = np.array(prectificados)
imgCopy = np.array(img)
print(puntos)
print(np.array(prectificados).T.astype(int).reshape((-1, 1, 2)))
H, _ = cv.findHomography(puntos, np.array(prectificados).reshape((-1, 1, 2)))

rectificado = cv.warpPerspective(imgCopy, H, (1280, 720))
cv.polylines(imgCopy, [np.array(preal)], True, (0, 0, 255), 2)
cv.imshow('Resultado', imgCopy)
rectificadoCpy = rectificado
for key, frame in autoStream():

    for p in points:
        rectificado = np.array(rectificadoCpy)
        cv.circle(rectificado, p, 3, (0, 255, 0), -1)

    if len(points) == 2:
        cv.line(rectificado, points[0], points[1], (0, 255, 0))

        # Calcula la media por columnas de los puntos.
        # Es decir, calcula el punto medio entre dos puntos
        # Calcula la media en el eje 0.
        c = np.mean(points, axis=0).astype(int)  # el astype(int) es para putText
        d = np.linalg.norm(np.array(points[0]) - points[1])

        # Calcular la distancia entre los puntos

        tam = (d / escala)

        points = deque(maxlen=2)

        putText(rectificado, f'{tam:.1f} ', c)

    cv.imshow('RECTIF', rectificado)


cv.destroyAllWindows()
