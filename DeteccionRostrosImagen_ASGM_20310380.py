# Areli Sarai García Medina | 20310380

import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('personas.jpg')

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Definir el clasificador Haar para detección de objetos
cascade_clasificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detectar objetos en la imagen
objects = cascade_clasificador.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

# Dibujar las ventanas alrededor de los objetos detectados
for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Mostrar la imagen con las ventanas alrededor de los objetos detectados
cv2.imshow('Imagen', img)
cv2.waitKey(0)
cv2.destroyAllWindows()