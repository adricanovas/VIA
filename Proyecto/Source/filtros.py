import numpy as np
from cv2 import cv2 as cv

from umucv.util import ROI, putText, Video
from umucv.stream import autoStream

# Constantes
VENTANA_PRINCIPAL = 'VENTANA PRINCIPAL'
VENATANA_FILTROS = 'PANEL DE FILTROS'
cv.namedWindow(VENTANA_PRINCIPAL)
cv.moveWindow(VENTANA_PRINCIPAL, 0, 0)


def nada(arg):
    pass


FILTRO_ACTIVO = 0  # 0 - 4
TAM = 200


# Esta funcion se encarga de crear una ventana con las trackbars para cada uno de los filtros
def crearVentanaFiltros(filtro):
    if filtro == 0:
        pass
    # Gaussiano
    if filtro == 1:
        cv.namedWindow(VENATANA_FILTROS)
        cv.createTrackbar("Fuerza del suavizado", VENATANA_FILTROS, 1, 100, nada)
        frame = np.zeros((TAM, TAM, 3), np.uint8)

        cv.imshow(VENATANA_FILTROS, frame)
    # BoxFilter
    if filtro == 2:
        cv.namedWindow(VENATANA_FILTROS)
        cv.createTrackbar("Centro del kernel", VENATANA_FILTROS, 1, 100, nada)
        frame = np.zeros((TAM, TAM, 3), np.uint8)

        cv.imshow(VENATANA_FILTROS, frame)
    # MedianBlur
    if filtro == 3:
        cv.namedWindow(VENATANA_FILTROS)
        cv.createTrackbar("Tamaño del kernel", VENATANA_FILTROS, 1, 100, nada)
        frame = np.zeros((TAM, TAM, 3), np.uint8)

        cv.imshow(VENATANA_FILTROS, frame)
    # bilateralFilter
    if filtro == 4:
        cv.namedWindow(VENATANA_FILTROS)
        cv.createTrackbar("Influencia de los colores", VENATANA_FILTROS, 1, 100, nada)
        cv.createTrackbar("Influencia de los pixeles", VENATANA_FILTROS, 1, 100, nada)
        frame = np.zeros((TAM, TAM, 3), np.uint8)

        cv.imshow(VENATANA_FILTROS, frame)


cv.createTrackbar('Filtro seleccionado', VENTANA_PRINCIPAL, 0, 4, nada)

region = ROI(VENTANA_PRINCIPAL)

for key, frame in autoStream():
    if key == ord('q'):
        break
    if FILTRO_ACTIVO != cv.getTrackbarPos('Filtro seleccionado', VENTANA_PRINCIPAL):
        cv.destroyWindow(VENATANA_FILTROS)

        crearVentanaFiltros(cv.getTrackbarPos('Filtro seleccionado', VENTANA_PRINCIPAL))
        FILTRO_ACTIVO = cv.getTrackbarPos('Filtro seleccionado', VENTANA_PRINCIPAL)

    if region.roi:
        # Convertimos en escala de grises (no es obligatorio)
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Aplicamos suavizado para eliminar ruido

        # gaus = cv.GaussianBlur(gray, (21, 21), cv.BORDER_DEFAULT)
        [rxi, ry1, rxf, ryf] = region.roi  # Coords de la zona señalada
        roiFrame = frame[ry1:ryf + 1, rxi:rxf + 1]

        # Aplicamos el filtro correspondiente
        if FILTRO_ACTIVO == 1:
            intensidad = cv.getTrackbarPos("Fuerza del suavizado", VENATANA_FILTROS)
            if not intensidad:
                intensidad = 1
            roiFrame = cv.GaussianBlur(roiFrame, (0, 0), intensidad)
        elif FILTRO_ACTIVO == 2:
            intensidad = cv.getTrackbarPos("Centro del kernel", VENATANA_FILTROS)
            if not intensidad:
                intensidad = 1
            roiFrame = cv.boxFilter(roiFrame, -1, (intensidad, intensidad))
        elif FILTRO_ACTIVO == 3:
            intensidad = cv.getTrackbarPos("Tamaño del kernel", VENATANA_FILTROS)
            if not intensidad % 2:
                intensidad = intensidad + 1
            roiFrame = cv.medianBlur(roiFrame, intensidad)
        elif FILTRO_ACTIVO == 4:
            intensidad = cv.getTrackbarPos("Influencia de los colores", VENATANA_FILTROS)
            intensidad1 = cv.getTrackbarPos("Influencia de los pixeles", VENATANA_FILTROS)
            if not intensidad:
                intensidad = 1
            if not intensidad1:
                intensidad1 = 1
            roiFrame = cv.bilateralFilter(roiFrame, 0, intensidad, intensidad1)
        # Lo dibujamos
        frame[ry1:ryf + 1, rxi:rxf + 1] = roiFrame
        cv.rectangle(frame, (rxi, ry1), (rxf, ryf), color=(0, 255, 255), thickness=2)
        if FILTRO_ACTIVO == 0:
            pass
        if FILTRO_ACTIVO == 1:
            putText(frame, "Gaussian Blur", orig=(rxi, ry1 - 8))
        if FILTRO_ACTIVO == 2:
            putText(frame, "Box Filter", orig=(rxi, ry1 - 8))
        if FILTRO_ACTIVO == 3:
            putText(frame, "Median Blur", orig=(rxi, ry1 - 8))
        if FILTRO_ACTIVO == 4:
            putText(frame, "Bilateral Filter", orig=(rxi, ry1 - 8))

    putText(frame, "0 - Ninguno", orig=(5, 16))
    putText(frame, "1 - Gaussian Blur", orig=(5, 36))
    putText(frame, "2 - Box Filter", orig=(5, 56))
    putText(frame, "3 - Median Blur", orig=(5, 76))
    putText(frame, "4 - Bilateral Filter", orig=(5, 96))
    cv.imshow(VENTANA_PRINCIPAL, frame)

cv.destroyAllWindows()
