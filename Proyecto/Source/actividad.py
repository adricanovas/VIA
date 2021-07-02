import numpy as np
from cv2 import cv2 as cv

from umucv.util import ROI, putText, Video
from umucv.stream import autoStream

# Constantes
THRESHOLD = 25
AREA_MIN = 500
GAUSSIANSMOOTH = 0

POS_CIR = 25
RAD_CIR = 15
COL_CIR = (0, 0, 255)
MAX_THRES = 255
TODOS_RECUADROS = 0
VENTANA_PRINCIPAL = "Actividad"

VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
AZUL = (255, 0, 0)

# Preapramos el video
video = Video(fps=30)
video.ON = True

cv.namedWindow(VENTANA_PRINCIPAL)
cv.moveWindow(VENTANA_PRINCIPAL, 0, 0)
region = ROI(VENTANA_PRINCIPAL)

fondo = None
for key, frame in autoStream():
    # Si pulsamos la q cerramos la aplicacion
    if key == ord('q'):
        exit(0)
    frameh, framew, _ = frame.shape
    frameCopy = np.array(frame)

    frameObjetosDetectados = np.zeros((frameh, framew, 3), np.uint8);
    if not region.roi and fondo is not None:
        fondo = None
    elif region.roi:
        [xi, yi, xf, yf] = region.roi  # Coords de la zona señalada

        # pasamos a monocromo
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Aplicamos suavizado para eliminar ruido ya que asi podremos obtener mejor las diferencias
        suavizadoGaussiano = cv.GaussianBlur(gray, (0, 0), 1)
        cv.rectangle(frame, (xi, yi), (xf, yf), color=(0, 255, 255), thickness=2)
        # Si no hay ningun fondo guardamos el fondo inicial con el que vamos a realizar las comparaciones
        if fondo is None:
            fondo = suavizadoGaussiano

        # Calculamos la diferencia entre el background y la imagen actual
        diff = cv.absdiff(fondo, suavizadoGaussiano)
        _, thresh = cv.threshold(diff, THRESHOLD, MAX_THRES, cv.THRESH_BINARY)
        dilatados = cv.dilate(thresh, None, iterations=2)
        redilatados = cv.dilate(dilatados, None, iterations=4)
        cv.imshow('thresh', dilatados)
        contours_raw, _ = cv.findContours(dilatados.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for c in contours_raw:
            bxi, byi, w, h = cv.boundingRect(c)
            bxf = bxi + w
            byf = byi + h
            # Eliminamos los contornos más pequeños( los que no tienen un area superior a AREA_MIN)
            if cv.contourArea(c) < AREA_MIN:
                if TODOS_RECUADROS:
                    cv.rectangle(frame, (bxi, byi), (bxf, byf), ROJO, 2)
                continue
            if (xf < bxi) or (xi > bxf):
                if TODOS_RECUADROS:
                    cv.rectangle(frame, (bxi, byi), (bxf, byf), AZUL, 2)
                continue
            elif (yf < byi) or (yi > byf):
                if TODOS_RECUADROS:
                    cv.rectangle(frame, (bxi, byi), (bxf, byf), AZUL, 2)
                continue

            valid_contours.append((bxi, byi, w, h))

            frameObjetosDetectados[byi:byf + 1, bxi:bxf + 1, 0] = (redilatados[byi:byf + 1,
                                                                   bxi:bxf + 1] / 255) * frameCopy[byi:byf + 1,
                                                                                         bxi:bxf + 1, 0]
            frameObjetosDetectados[byi:byf + 1, bxi:bxf + 1, 1] = (redilatados[byi:byf + 1,
                                                                   bxi:bxf + 1] / 255) * frameCopy[byi:byf + 1,
                                                                                         bxi:bxf + 1, 1]
            frameObjetosDetectados[byi:byf + 1, bxi:bxf + 1, 2] = (redilatados[byi:byf + 1,
                                                                   bxi:bxf + 1] / 255) * frameCopy[byi:byf + 1,
                                                                                         bxi:bxf + 1, 2]

            cv.rectangle(frame, (bxi, byi), (bxf, byf), (0, 255, 0), 2)
        ## OPCIONAL:

        if valid_contours:
            cv.circle(frame, (POS_CIR, POS_CIR), RAD_CIR, COL_CIR, -1)
            video.write(frame)

        # Solo si hay un roi seleccionado podemos borrarlo o cambiar el fondo
        if key == ord('x'):
            region.roi = []
        if key == ord('f'):
            fondo = None

    cv.imshow(VENTANA_PRINCIPAL, frame)
    cv.imshow("Objetos detectados", frameObjetosDetectados)
cv.destroyAllWindows()

video.release()
