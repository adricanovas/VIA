#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText
import sys

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    sys.exit("Expecting a single image file argument")

image = cv.imread(filename)
print(image.shape)

image_small = cv.resize(image, (800, 600))

textColor = (0, 0, 255)  # red
cv.putText(image_small, "Hello World!!!", (200, 200),
           cv.FONT_HERSHEY_PLAIN, 3.0, textColor,
           thickness=4)
# .....
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
        cv.line(frame, points[0],points[1],(0,0,255))
        c = np.mean(points, axis=0).astype(int)
        d = np.linalg.norm(np.array(points[1])-points[0])
        putText(frame,f'{d:.1f} pix',c)

    cv.imshow('webcam',frame)

cv.imshow('Hello World GUI', image_small)
cv.waitKey()
cv.destroyAllWindows()
