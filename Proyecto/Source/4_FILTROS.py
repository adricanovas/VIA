import numpy as np
from cv2 import cv2 as cv

from umucv.util import ROI, putText, Video
from umucv.stream import autoStream

cv.namedWindow('Filtros')
cv.moveWindow('Filtros', 0, 0)

filtro = 0
region = ROI('Filtros')

for key, frame in autoStream():
    if key == ord('0'):
        # NINGUNO
        filtro = 0
    if key == ord('1'):
        # LAPLACIANO
        filtro = 1
    if key == ord('2'):
        # GAUSSIANO
        kSize = 2
        filtro = 2
    if key == ord('3'):
        # MEDIAN
        kSize = 1
        filtro = 3
    if key == ord('4'):
        # BILATERAL
        diam = 0
        sColor = 5
        sSpace = 0
        filtro = 4
    # Operadores morfológicos
    if key == ord('5'):
        # EROSION
        kSize = 5
        filtro = 5
    if key == ord('6'):
        # DILATACION
        kSize = 5
        filtro = 6
    if key == ord('7'):
        # Black Hat
        filtro = 7
        kSize = 5
    if (filtro == 0):
        None
    if region.roi:
        [x1, y1, x2, y2] = region.roi
        roiFrame = frame[y1:y2 + 1, x1:x2 + 1]

        # Aplicación de filtros
        if filtro == 0:
            pass
        if filtro == 1:
            # LAPLACIANO¡
            putText(frame, "LAPLACIANO", orig=(x1, y1))
            roiFrame = cv.Laplacian(roiFrame, -1)
        if filtro == 2:
            # GAUSSIANO
            putText(frame, "GAUSSIANO", orig=(x1, y1))
            putText(frame, 'Intensidad: ' + str(kSize), orig=(5, 15))
            if (key == ord('m')):
                kSize = kSize + 1
            roiFrame = cv.GaussianBlur(roiFrame, (0, 0), kSize)
        if filtro == 3:
            # MEDIAN
            putText(frame, "MEDIA", orig=(x1, y1))
            putText(frame, 'Intensidad: ' + str(kSize), orig=(5, 15))
            if (key == ord('m')):
                kSize = kSize + 2
            roiFrame = cv.medianBlur(roiFrame, kSize)
        if filtro == 4:
            # BILATERAL
            putText(frame, 'Diametro: ' + str(diam), orig=(5, 15))
            putText(frame, 'sigmaColor: ' + str(sColor), orig=(5, 35))
            putText(frame, 'sigmaSpace: ' + str(sSpace), orig=(5, 55))
            if (key == ord('m')):
                diam = diam + 1
            if (key == ord('n')):
                sColor = sColor + 1
            if (key == ord('b')):
                sSpace = sSpace + 1
            putText(frame, "BILATERAL", orig=(x1, y1))
            roiFrame = cv.bilateralFilter(roiFrame, diam, sColor, sSpace)
        if filtro == 5:
            # EROSION
            putText(frame, "EROSION", orig=(x1, y1))
            putText(frame, 'kSize: ' + str(kSize), orig=(5, 15))
            if (key == ord('m')):
                kSize = kSize + 1
            kernel = np.ones((kSize, kSize), np.uint8)
            roiFrame = cv.erode(roiFrame, kernel, iterations=1)
        if filtro == 6:
            # DILATACION
            putText(frame, "DILATACION", orig=(x1, y1))
            putText(frame, 'kSize: ' + str(kSize), orig=(5, 15))
            if (key == ord('m')):
                kSize = kSize + 1
            kernel = np.ones((kSize, kSize), np.uint8)
            roiFrame = cv.dilate(roiFrame, kernel, iterations=1)
        if filtro == 7:
            # Black Hat
            putText(frame, "Black Hat", orig=(x1, y1))
            putText(frame, 'kSize: ' + str(kSize), orig=(5, 15))
            if (key == ord('m')):
                kSize = kSize + 1
            kernel = np.ones((kSize, kSize), np.uint8)
            roiFrame = cv.morphologyEx(roiFrame, cv.MORPH_BLACKHAT, kernel)

        frame[y1:y2 + 1, x1:x2 + 1] = roiFrame
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
    cv.imshow('Filtros', frame)
cv.destroyAllWindows()