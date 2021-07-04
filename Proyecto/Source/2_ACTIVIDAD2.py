import numpy as np
from cv2 import cv2 as cv
from umucv.util import ROI, putText, Video

cap = cv.VideoCapture(0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.createBackgroundSubtractorMOG2()

cv.namedWindow('Actividad')
cv.moveWindow('Actividad', 0, 0)
region = ROI('Actividad')

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    bg = cv.bitwise_and(frame, frame, mask=fgmask)
    final = cv.bitwise_or(bg, cv.resize(background, (1280, 720)))


    cv.imshow('frame', fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    cv.imshow('Actividad', bg)
cap.release()
cv.destroyAllWindows()