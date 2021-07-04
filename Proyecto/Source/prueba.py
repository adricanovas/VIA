from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from umucv.stream import autoStream
from umucv.util import ROI, putText

# Contantes
MAIN_WINDOW = 'Principal'
MODELOS = []
LIMITE = 0.7
THICKNESS = 2
RESIZE_TAM = 160
EJEY = 10000
VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
AZUL = (255, 0, 0)

TOMAVALORES = np.arange(257)
OPTIONAL_GAUS = 10


def nada(v):
    pass


cv.namedWindow(MAIN_WINDOW)
# Creacion de las trackbars
cv.createTrackbar("Gauss", MAIN_WINDOW, 10, 50, nada)
cv.createTrackbar("Limite", MAIN_WINDOW, 10, 50, nada)
cv.createTrackbar("Anchura lineas del histograma", MAIN_WINDOW, 2, 5, nada)
cv.createTrackbar("Tamaño Miniaturas", MAIN_WINDOW, 160, 300, nada)
cv.createTrackbar("Ajuste de escala del histograma", MAIN_WINDOW, 10000, 100000, nada)

cv.moveWindow(MAIN_WINDOW, 0, 0)
region = ROI(MAIN_WINDOW)


def rgb2yuv(x):
    return cv.cvtColor(x, cv.COLOR_RGB2YUV)


# Calcula el histograma (normalizado) de los canales conjuntos UV
def uvh(x):
    # Normalizar un histograma
    # para tener frecuencias que sumaran 1
    # en vez de número de elementos
    def normhist(x):
        return x / np.sum(x)

    yuv = rgb2yuv(x)
    h = cv.calcHist([yuv],  # necesario ponerlo en una lista aunque solo admite un elemento
                    [1, 2],  # elegimos los canales U y V
                    None,  # posible máscara
                    [32, 32],  # las cajitas en cada dimensión
                    [0, 256] + [0, 256])  # rango de interés (todo)
    return normhist(h)


def apartadoOpcional(img):
    med = [np.mean(r, (0, 1)) for r in MODELOS]

    hist = [uvh(i) for i in MODELOS]
    # Canales UY reducidos a una resolución de 5 bits (32 niveles)
    uvr = np.floor_divide(cv.cvtColor(img, cv.COLOR_RGB2YUV)[:, :, [1, 2]], 8)
    u = uvr[:, :, 0]
    v = uvr[:, :, 1]
    lik = [h[u, v] for h in hist]
    # Se suaviza un poco para hacer que los pixels vecinos influyan un poco en los casos dudosos.
    lik = [cv.GaussianBlur(i, (0, 0), OPTIONAL_GAUS) for i in lik]
    E = np.sum(lik, axis=0)
    p = np.array(lik) / E
    c = np.argmax(p, axis=0)
    mp = np.max(p, axis=0)
    mp[E < 0.1] = 0

    res = np.zeros(img.shape, np.uint8)
    for k in range(len(MODELOS)):
        res[c == k] = med[k]

    cv.imshow('Apartado Opcional', res)

    '''
    res[mp < 0.9] = 128, 0, 0
    res[E < 0.05] = 0, 0, 0
    cv.imshow('Apartado', res)
    '''


# Crea un diagrama en una region definida del plano
def getValoresXYZ(b, h, ryf, ryi, rxf, rxi):
    altura = ryf - ryi
    anchura = rxf - rxi
    xs = b[1:] * (anchura / 255)
    ys = altura - h * (altura / (EJEY + 1))
    print(xs.shape)
    print(ys.shape)
    xys = np.array([xs, ys]).T.astype(int)
    return xys


for key, frame in autoStream():
    texto = ' '
    # Cargamos los valores que contengan las trackbars
    LIMITE = cv.getTrackbarPos("Limite", MAIN_WINDOW)
    AREA_MIN = cv.getTrackbarPos("Anchura lineas del histograma", MAIN_WINDOW)
    RESIZE_TAM = cv.getTrackbarPos("Tamaño Miniaturas", MAIN_WINDOW)
    EJEY = cv.getTrackbarPos("Ajuste de escala del histograma", MAIN_WINDOW)
    OPTIONAL_GAUS = cv.getTrackbarPos("Gauss", MAIN_WINDOW)
    if OPTIONAL_GAUS == 0:
        OPTIONAL_GAUS = 1;
    if MODELOS:
        apartadoOpcional(frame)
    if region.roi:
        # Seleccionamos la region del roy hacemos una copia
        [rxi, ryi, rxf, ryf] = region.roi
        seleccionE = frame[ryi:ryf + 1, rxi:rxf + 1]
        seleccion = np.array(seleccionE)
        seleccion_copy = np.array(seleccion)

        cv.rectangle(frame, (rxi, ryi), (rxf, ryf), color=(0, 255, 255), thickness=2)

        blue = seleccion[:, :, 0]
        green = seleccion[:, :, 1]
        red = seleccion[:, :, 2]

        # Creamos cada uno de los histogramas para cada uno de los colores
        bh, bb = np.histogram(blue, TOMAVALORES)
        cv.polylines(seleccionE, [getValoresXYZ(bb, bh, ryf, ryi, rxf, rxi)], isClosed=False, color=AZUL,
                     thickness=THICKNESS)
        bh = cv.normalize(bh, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        gh, gb = np.histogram(green, TOMAVALORES)
        cv.polylines(seleccionE, [getValoresXYZ(gb, gh, ryf, ryi, rxf, rxi)], isClosed=False, color=VERDE,
                     thickness=THICKNESS)
        gh = cv.normalize(gh, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        rh, rb = np.histogram(red, TOMAVALORES)
        cv.polylines(seleccionE, [getValoresXYZ(rb, rh, ryf, ryi, rxf, rxi)], isClosed=False, color=ROJO,
                     thickness=THICKNESS)
        rh = cv.normalize(rh, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        # Agregamos,usando la seleccion actual a los modelos y posteriormente imprimimos los modelos (reescalados)
        if key == ord('p'):
            MODELOS.append(seleccion_copy)
        if MODELOS:
            aux = []
            for img in MODELOS:
                reescalado = cv.resize(img, (RESIZE_TAM, RESIZE_TAM), interpolation=cv.INTER_AREA)
                aux.append(reescalado)
            mixed_frames = np.hstack(aux)
            cv.imshow('SAVED', mixed_frames)

            # Vemos como se ajusta a cada uno de los modelos
        parecidos = []
        valores = []
        for i in MODELOS:
            bluei = i[:, :, 0]
            greeni = i[:, :, 1]
            redi = i[:, :, 2]
            # Creamos cada uno de los histogramas para cada uno de los colores
            bhi, bbi = np.histogram(bluei, TOMAVALORES)
            bhi = cv.normalize(bhi, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            ghi, gbi = np.histogram(greeni, TOMAVALORES)
            ghi = cv.normalize(ghi, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            rhi, rbi = np.histogram(redi, TOMAVALORES)
            rhi = cv.normalize(rhi, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            # Comparamos el modelo con el actual y lo metemos en un lista al estilo de un diccionario, donde otra lista ira relacionada a esta usando el parecido como clave
            difB = cv.absdiff(bhi, bh)
            difG = cv.absdiff(ghi, gh)
            difR = cv.absdiff(rhi, rh)

            sumB = np.sum(difB)
            sumG = np.sum(difG)
            sumR = np.sum(difR)

            maxTotal = np.amax([sumB, sumG, sumR])
            parecidos.append(i)
            valores.append(maxTotal)
        # Si no tiene parecidos entonces la imagen que tenemos que poner es una negra
        if not parecidos:
            cv.imshow('COINCIDENCE', np.zeros((RESIZE_TAM, RESIZE_TAM, 3), np.uint8))
        else:
            # Si tiene entonces buscamos el que tenga el parecido menor
            minParecido = np.amin(valores)
            if minParecido > LIMITE:
                cv.imshow('COINCIDENCE', np.zeros((RESIZE_TAM, RESIZE_TAM, 3), np.uint8))
            else:
                reescalado = cv.resize(parecidos[valores.index(minParecido)], (RESIZE_TAM, RESIZE_TAM),
                                       interpolation=cv.INTER_AREA)
                cv.imshow('COINCIDENCE', reescalado)
        # Esto ultimo lo usamos para imprimir los numeros en la pantalla
        rounded = [round(f, 2) for f in valores]
        stringList = [str(r) for r in rounded]
        texto = ' '.join(stringList)
    # Si para salir del ciclo
    if key == ord('q'):
        break

    # plt.pause(0.001)
    # fig.canvas.draw_idle()
    # fig.canvas.start_event_loop(0.001)
    if texto:
        putText(frame, texto)

    cv.imshow(MAIN_WINDOW, frame)

cv.destroyAllWindows()