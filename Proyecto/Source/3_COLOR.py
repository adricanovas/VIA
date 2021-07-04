# Ejemplo de reproyección de histograma

# Seleccionando un roi y pulsando c se captura el modelo de
# color de la región en forma de histograma y se muestra
# la verosimilitud de cada pixel de la imagen en ese modelo.

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream, mkStream

cv.namedWindow("original")
roi = ROI("original")

def hist(x, redu=16):
    return cv.calcHist([x],
                       [0, 1, 2],  # canales a considerar
                       None,  # posible máscara
                       [redu, redu, redu],  # número de cajas en cada canal
                       [0, 256] + [0, 256] + [0, 256])  # intervalo a considerar en cada canal


H = None
models = list()
for key, frame in autoStream():

    if H is not None:
        b, g, r = np.floor_divide(frame, 16).transpose(2, 0, 1)
        L = H[b, g, r]  # indexa el array H simultáneamente en todos los
        # pixels de la imagen.
        cv.imshow("likelihood", L / L.max())

    if roi.roi:
        [x1, y1, x2, y2] = roi.roi
        trozo = frame[y1:y2 + 1, x1:x2 + 1]
        # cv.imshow("trozo", trozo)
        H = hist(trozo)
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

        blue = trozo[:, :, 0].copy()
        green = trozo[:, :, 1].copy()
        red = trozo[:, :, 2].copy()

        # Histogramas del canal RED
        histBlue, binsBlue = np.histogram(blue, bins=32, range=(0, 257))
        histGreen, binsGreen = np.histogram(green, bins=32, range=(0, 257))
        histRed, binsRed = np.histogram(red, bins=32, range=(0, 257))
        # Histogramas del canal Blue
        xsRed = binsRed[1:]
        ysRed = 480 - histRed * (480 / 10000)
        xysRed = np.array([xsRed, ysRed]).T.astype(int)
        cv.polylines(trozo, [xysRed], isClosed=False, color=(0, 0, 255), thickness=2)
        # Histogramas del canal Green
        xsGreen = binsGreen[1:]
        ysGreen = 480 - histGreen * (480 / 10000)
        xysGreen = np.array([xsGreen, ysGreen]).T.astype(int)
        cv.polylines(trozo, [xysGreen], isClosed=False, color=(0, 255, 0), thickness=2)

        xsBlue = binsBlue[1:]
        ysBlue = 480 - histBlue * (480 / 10000)
        xysBlue = np.array([xsBlue, ysBlue]).T.astype(int)
        cv.polylines(trozo, [xysBlue], isClosed=False, color=(255, 0, 0), thickness=2)

        if key == ord('c'):
            models.append((trozo, histRed, histGreen, histBlue))
            aux = list()
            for m in models:
                scaled = cv.resize(np.array(m[0]), (160, 160), interpolation=cv.INTER_AREA)
                aux.append(scaled)
            mixed_frames = np.hstack(aux)
            cv.imshow('models', mixed_frames)
        if models:
            results = list()
            text = ""
            for m in models:
                [m1, hr, hg, hb] = m
                dR = np.sum(cv.absdiff(histRed, hr))
                dG = np.sum(cv.absdiff(histGreen, hg))
                dB = np.sum(cv.absdiff(histBlue, hb))
                results.append(max(dR, dG, dB) / 1000)
            bestResult = min(results)
            if results:
                text = text + str(results)
                putText(frame, text)
            if bestResult < 5.2:
                i = results.index(bestResult)
                cv.imshow("detected", models[i][0])
            else:
                cv.imshow('detected', np.zeros((160, 160, 3), np.uint8))
    cv.imshow('original', frame)
