import cv2 as cv
import numpy as np
import sys
import os
from glob import glob

models = list()
tamXMax = 0
tamYMax = 0
files = glob(sys.argv[1])

for f in files:
	img = cv.imread(f)
	models.append(img)

# Definimos la matriz homogénea de desplazamiento
def desp(d):
	dx, dy = d
	return np.array([
		[1, 0, dx],
		[0, 1, dy],
		[0, 0, 1]])

# Función auxiliar para llevar las imágenes a un marco común
def t(h, x, num):
	if num == 0:
		return cv.warpPerspective(x, desp((0, tamYMax//2)) @ h, (tamXMax, tamYMax))
	if num == 1:
		return cv.warpPerspective(x, desp((tamXMax-(tamXMax//3), tamYMax//2)) @ h, (tamXMax, tamYMax))
	else:
		return cv.warpPerspective(x, desp((0, tamYMax//4)) @ h, (tamXMax+300, tamYMax+300))


def unir(img1, img2, num):
	global tamYMax, tamXMax
	y, x = img1.shape[:2]
	yi, xi = img2.shape[:2]
	tamXMax = x + xi
	tamYMax = y + yi

	sift = cv.xfeatures2d.SIFT_create()
	if num == 0 or num == 2:
		(kps1, descs1) = sift.detectAndCompute(img1, None)
		(kps2, descs2) = sift.detectAndCompute(img2, None)
	if num == 1:
		(kps1, descs1) = sift.detectAndCompute(img2, None)
		(kps2, descs2) = sift.detectAndCompute(img1, None)

	bf = cv.BFMatcher()
	matches = bf.knnMatch(descs1, descs2, k = 2)
	good = []
	for m in matches:
		if len(m) == 2:
			best, second = m
			if best.distance < 0.75 * second.distance:
				good.append(best)

	src_pts = np.array([kps2[m.trainIdx].pt for m in good]).astype(np.float32).reshape(-1, 2)
	dst_pts = np.array([kps1[m.queryIdx].pt for m in good]).astype(np.float32).reshape(-1, 2)

	H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)
	if num == 0:
		new = np.maximum(t(np.eye(3), img1, 0), t(H, img2, 0))
	if num == 1:
		new = np.maximum(t(np.eye(3), img2, 1), t(H, img1, 1))
	if num == 2:
		new = np.maximum(t(np.eye(3), img1, 2), t(H, img2, 2))

	return new

def panoramica(models):
	# Seleccionamos los modelos que van en las posiciones
	longitud = len(models)
	middle = longitud//2
	centro = models[middle - 1]
	izquierda = models[0]
	derecha = models[longitud - 1]
	contIzq = 0
	contDer = longitud-1
	# Unimos con los de la derecha
	while contDer >= middle:
		derecha = unir(derecha, models[contDer], 1)
		contDer -= 1
	# Unimos con los de la izquierda
	while contIzq < middle - 1:
		izquierda = unir(models[contIzq], izquierda, 0)
		contIzq += 1

	nueva = unir(centro, izquierda, 0)

	nueva = unir(derecha, nueva, 2)
	return nueva


model = cv.resize(np.hstack(models), (1000, 360))
cv.imshow('Models', model)


while True:
	key = cv.waitKey(1) & 0xFF
	if key == 27: break
	if key == ord('s'):
		imagen = panoramica(models)
		cv.imshow('PANO', cv.resize(imagen, (1280, 720)))
