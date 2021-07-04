# María José Navarro Moreno
# Ejercicio 12
# ViA 2017

import cv2 as cv
import numpy as np
import sys
import os
from glob import glob

models = [] # contiene las imagenes a leer concatenadas
tamXMax = 0
tamYMax = 0


def tamMax(x, y, xi, yi):
	global tamYMax, tamXMax
	tamXMax = x+xi
	tamYMax = y+yi


def desp(d):
	dx, dy = d
	return np.array([
		[1, 0, dx],
		[0, 1, dy],
		[0, 0, 1]])


def t(h, x, num):
	global tamXMax, tamYMax
	if num == 0: # Izquierda
		return cv.warpPerspective(x, desp((0, tamYMax//2)) @ h, (tamXMax, tamYMax))
	if num == 1: # Derecha
		return cv.warpPerspective(x, desp((tamXMax-(tamXMax//3), tamYMax//2)) @ h, (tamXMax, tamYMax))
	else: # union final
		return cv.warpPerspective(x, desp((0, tamYMax//4)) @ h, (tamXMax+300, tamYMax+300))


def unir(img1, img2, num):
	sift = cv.xfeatures2d.SIFT_create()
	if num == 0 or num == 2: # Izquierda
		(kps1, descs1) = sift.detectAndCompute(img1, None)
		(kps2, descs2) = sift.detectAndCompute(img2, None)
	if num == 1: # derecha
		(kps1, descs1) = sift.detectAndCompute(img2, None)
		(kps2, descs2) = sift.detectAndCompute(img1, None)

	bf = cv.BFMatcher()
	matches = bf.knnMatch(descs1, descs2, k = 2)
	good = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good.append(m)

	src_pts = np.array([kps2[m.trainIdx].pt for m in good]).astype(np.float32).reshape(-1, 2)
	dst_pts = np.array([kps1[m.queryIdx].pt for m in good]).astype(np.float32).reshape(-1, 2)

	y, x = img1.shape[:2]
	yi, xi = img2.shape[:2]
	tamMax(x, y, xi, yi)

	H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)
	if num == 0: # Izquierda
		new = np.maximum(t(np.eye(3), img1, 0), t(H, img2, 0))
	if num == 1: # derecha
		new = np.maximum(t(np.eye(3), img2, 1), t(H, img1, 1))
	if num == 2: # union final
		new = np.maximum(t(np.eye(3), img1, 2), t(H, img2, 2))

	return new


def repart(models, longitud):
	if longitud%2 != 0:
		middle = 1+longitud//2
	else:
		middle = longitud//2
	if longitud == 3:
		centro = models[1]
		izquierda = models[0]
		derecha = models[2]
	else:
		centro = models[middle - 1] # Menos uno porque los arrays empiezan en cero
		izquierda = models[0] # La primera img
		derecha = models[longitud - 1] # La ultima

	contIzq = 0
	contDer = longitud-1
	return centro, izquierda, derecha,  contIzq, contDer, middle


def panoramica(models):
	longitud = len(models)
	if longitud == 1:
		return models[0]
	elif longitud == 2:
		return unir(models[1], models[0], 0)

	centro, izquierda, derecha, contIzq, contDer, middle = repart(models, longitud)
	# El contador de la izquierda tiene que llegar a la mitad y el de la derecha al final
	while contDer >= middle:
		derecha = unir(derecha, models[contDer], 1)
		contDer -= 1

	while contIzq < middle - 1:
		izquierda = unir(models[contIzq], izquierda, 0)
		contIzq += 1

	nueva = unir(centro, izquierda, 0)

	nueva = unir(derecha, nueva, 2)
	return nueva

print("""
Pulse s para crear la panoramica
    """)
files = glob(sys.argv[1])
models = list()

for fn in files:
	img = cv.imread(fn)
	models.append(img)

img4 = cv.resize(np.hstack(models), (1000, 360))
cv.imshow('Muestra', img4)
while True:
	key = cv.waitKey(1) & 0xFF
	if key == 27: break
	if key == ord('s'):
		imagen = panoramica(models)
		cv.imshow('Ejr. 12', cv.resize(imagen, (1000, 400)))
