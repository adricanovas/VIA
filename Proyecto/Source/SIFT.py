#!/usr/bin/env python

# eliminamos muchas coincidencias erróneas mediante el "ratio test"

import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText
from collections import deque



sift = cv.AKAZE_create()

matcher = cv.BFMatcher()

images = deque(maxlen=20)

for key, frame in autoStream():

    if key == ord('x'):
        x0 = None

    t0 = time.time()
    keypoints, descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000 * (t1 - t0):.0f} ms')

    x0 = cv.imread("../images/SIFT/1.jpg", 1)
    t2 = time.time()
    # solicitamos las dos mejores coincidencias de cada punto, no solo la mejor
    k0, d0 = sift.detectAndCompute(x0, mask=None)
    matches = matcher.knnMatch(descriptors, d0, k=2)
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

    imgm = cv.drawMatches(frame, keypoints, x0, k0, good,
                          flags=0,
                          matchColor=(128, 255, 128),
                          singlePointColor=(128, 128, 128),
                          outImg=None)

    putText(imgm, f'{len(good)} matches  {1000 * (t3 - t2):.0f} ms',
            orig=(5, 36), color=(200, 255, 200))
    cv.imshow("SIFT", imgm)