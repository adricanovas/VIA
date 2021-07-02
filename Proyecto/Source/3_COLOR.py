import numpy as np
from cv2 import cv2 as cv
from umucv.stream import autoStream
from umucv.util import ROI, putText

# Contantes
MODELOS = []
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)


cv.namedWindow('Color')
cv.moveWindow('Color', 0, 0)
region = ROI('Color')

# Calcula el histograma (normalizado) de los canales conjuntos UV
def uvh(x):
    # Normalizar un histograma
    # para tener frecuencias que sumaran 1
    # en vez de número de elementos
    def normhist(x):
        return x / np.sum(x)

    yuv = cv.cvtColor(x, cv.COLOR_RGB2YUV)
    h = cv.calcHist([yuv],  # necesario ponerlo en una lista aunque solo admite un elemento
                    [1, 2],  # elegimos los canales U y V
                    None,  # posible máscara
                    [32, 32],  # las cajitas en cada dimensión
                    [0, 256] + [0, 256])  # intervalo a considerar en cada canal
    return normhist(h)

def apartadoOpcional(img):
    med = [np.mean(r, (0, 1)) for r in MODELOS]
    hist = [uvh(i) for i in MODELOS]
    # Canales UY reducidos a una resolución de 5 bits (32 niveles)
    uvr = np.floor_divide(cv.cvtColor(img, cv.COLOR_RGB2YUV)[:, :, [1, 2]], 8)
    u = uvr[:, :, 0]
    v = uvr[:, :, 1]
    lik = [h[u, v] for h in hist]
    # Utiliza un suavizado Gaussiano
    lik = [cv.GaussianBlur(i,(0,0), 1) for i in lik]
    E = np.sum(lik, axis=0)
    p = np.array(lik) / E
    c = np.argmax(p, axis=0)
    mp = np.max(p, axis=0)
    mp[E < 0.1] = 0

    res = np.zeros(img.shape, np.uint8)
    for k in range(len(MODELOS)):
        res[c == k] = med[k]
    cv.imshow('Apartado Opcional', res)

for key, frame in autoStream():
    texto = ' '
    if MODELOS:
        apartadoOpcional(frame)
    if region.roi:
        # Obtenemos las coordenadas de la región de interes

        [x1,y1,x2,y2] = region.roi  # Coords de la zona señalada
        region.roi = []
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        trozo = frame[y1:y2 + 1, x1:x2 + 1]

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

        blue = trozo.array(trozo)[:, :, 0]
        green = trozo.array(trozo)[:, :, 1]
        red = trozo.array(trozo)[:, :, 2]

        # agrupamos los niveles de gris en intervalos de ancho 1
        blue_h, blue_b = np.histogram(blue, np.arange(0, 257, 1))
        # cv.polylines(x,[pts],True, (0,128,255),2)
        # cv.polylines(gray, [xys], isClosed=False, color=0, thickness=2)
        # ajustamos la escala del histograma para que se vea bien en la imagen
        # usaremos cv.polylines, que admite una lista de listas de puntos x,y enteros
        # las coordenadas x son los bins del histograma (quitando el primero)
        # y las coordenadas y son el propio histograma escalado y desplazado
        xs = 2 * blue_b[1:]
        ys = 480 - blue_h * (480 / 100000)
        # trasponemos el array para emparejar cada x e y
        xys = np.array([xs, ys]).T.astype(int)
        cv.polylines(trozo, [xys], isClosed=False, color=(0, 0, 255), thickness=2)
        bh = cv.normalize(bh, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        gh, gb = np.histogram(green, TOMAVALORES)
        cv.polylines(seleccionE, [getValoresXYZ(gb, gh, ryf, ryi, rxf, rxi)], isClosed=False, color=VERDE,
                     thickness=2)
        gh = cv.normalize(gh, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        rh, rb = np.histogram(red, TOMAVALORES)
        cv.polylines(seleccionE, [getValoresXYZ(rb, rh, ryf, ryi, rxf, rxi)], isClosed=False, color=ROJO,
                     thickness=2)
        rh = cv.normalize(rh, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        # Agregamos,usando la seleccion actual a los modelos y posteriormente imprimimos los modelos (reescalados)
        if key == ord('p'):
            MODELOS.append(seleccion_copy)
        if MODELOS:
            aux = []
            for img in MODELOS:
                reescalado = cv.resize(img, (160, 160))
                aux.append(reescalado)
            mixed_frames = np.hstack(aux)
            cv.imshow('SAVED', mixed_frames)


        parecidos = []
        valores = []
        for i in MODELOS:
            bluei = i[:, :, 0]
            greeni = i[:, :, 1]
            redi = i[:, :, 2]

            bhi, bbi = np.histogram(bluei, TOMAVALORES)
            bhi = cv.normalize(bhi, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            ghi, gbi = np.histogram(greeni, TOMAVALORES)
            ghi = cv.normalize(ghi, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            rhi, rbi = np.histogram(redi, TOMAVALORES)
            rhi = cv.normalize(rhi, None, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

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
            if minParecido > 30:
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
        exit(0)

    putText(frame, texto)

    cv.imshow('Color', frame)

cv.destroyAllWindows()