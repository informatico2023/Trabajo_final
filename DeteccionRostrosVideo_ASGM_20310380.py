# Areli Sarai García Medina | 20310380

import cv2

# Cargar el clasificador Haar Cascade preentrenado para la detección de caras
cascada_cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)  # 0 representa la cámara web predeterminada, puedes cambiarlo si tienes múltiples cámaras

while True:
    # Leer el siguiente fotograma de la cámara
    ret, frame = cap.read()

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realizar la detección de caras utilizando el clasificador Haar Cascade
    caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de las caras detectadas
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el fotograma resultante
    cv2.imshow('Detección de caras', frame)

    # Salir si se presiona la tecla 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()