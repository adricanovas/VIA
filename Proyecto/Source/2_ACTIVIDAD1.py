from cv2 import cv2 as cv
from umucv.util import putText, Video
import sys

# Usamos tracker Discriminative Correlation Filter
tracker = cv.TrackerCSRT_create()

# Capturamos la cámara y la región ROI
video = cv.VideoCapture(0)
ok, frame = video.read()
bbox = cv.selectROI(frame, False)

# Iniciamos el tracker sobre el objeto
success = tracker.init(frame, bbox)

capture = Video(fps=20)
capture.ON = True
grabar = False
cv.destroyAllWindows()
while True:
    # Leemos frame a frame
    success, frame = video.read()
    success, bbox = tracker.update(frame)

    # Si tenemos trakeado el objeto
    if success:
        # Creamos la caja
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    if grabar:
        capture.write(frame)
    cv.imshow("Tracking", frame)
    k = cv.waitKey(1) & 0xff
    if k == ord('g'):
        grabar = not grabar
    elif k == 27:
        break
video.release()