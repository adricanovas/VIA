#!/usr/bin/env python

# eliminamos muchas coincidencias erróneas mediante el "ratio test"

import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText
from glob import glob

sift = cv.xfeatures2d.SIFT_create(nfeatures=200)

matcher = cv.BFMatcher()
threshold = 12
files = glob('../Images/5_SIFT/*.jpg')

# Cargamos las imagenes
imagenes = list()
for f in files:
    img = cv.imread(f)
    keypoints, descriptors = sift.detectAndCompute(img, mask=None)
    k0, d0, x0 = keypoints, descriptors, img
    imagenes.append((k0, d0, x0, f))

for key, frame in autoStream():

    t0 = time.time()
    keypoints, descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000 * (t1 - t0):.0f} ms')

    t2 = time.time()
    mejor = [-1, None]
    for v in imagenes:
        # solicitamos las dos mejores coincidencias de cada punto, no solo la mejor
        matches = matcher.knnMatch(descriptors, v[1], k=2)
        t3 = time.time()

        # ratio test
        # nos quedamos solo con las coincidencias que son mucho mejores que
        # que la "segunda opción". Es decir, si un punto se parece más o menos lo mismo
        # a dos puntos diferentes del modelo lo eliminamos.
        good = []
        for m in matches:
            if len(m) >= 2:
                best, second = m
                if best.distance < 0.75 * second.distance:
                    good.append(best)

        if (len(good) / len(v[0]) * 100) > threshold and (len(good) / len(v[0]) * 100) > mejor[0]:
            mejor[0] = (len(good) / len(v[0]) * 100)
            mejor[1] = v[2]

    if mejor[0] != -1:
        imgm = cv.drawMatches(frame, keypoints, x0, k0, good,
                              flags=0,
                              matchColor=(128, 255, 128),
                              singlePointColor=(128, 128, 128),
                              outImg=None)

        putText(imgm, f'{len(good)} matches  {1000 * (t3 - t2):.0f} ms',
                orig=(5, 36), color=(200, 255, 200))
        scale_percent = 10  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv.resize(mejor[1], dim, interpolation=cv.INTER_AREA)
        frame[200:200+height, 200:200+width, :] = resized[:, :, :]
    cv.imshow("5_SIFT", frame)
