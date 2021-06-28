#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText


points = deque(maxlen=2)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

for key, frame in autoStream():
    for p in points:
        cv.circle(frame, p,3,(0,0,255),-1)
    if len(points) == 2:
        cv.line(frame, points[0], points[1], (0, 0, 255))
        c = np.mean(points, axis=0).astype(int)
        d = np.linalg.norm(np.array(points[1]) - points[0])
        # Calculamos el valor del angulo
        FOV = (np.arctan((d / 2) / 1011.52938)) * 180 / np.pi
        FOV = round(FOV * 2, 2)
        putText(frame, f'{FOV}' + '^º', c)
    cv.imshow('webcam', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
            break
cv.destroyAllWindows()
