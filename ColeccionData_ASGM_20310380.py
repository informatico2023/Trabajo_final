# Areli Sarai García Medina | 20310380

import cv2
import os

video = cv2.VideoCapture(0)

# Cargar el clasificador Haar Cascade preentrenado para la detección de caras
cascada_cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

nombreID = str(input("Ingresa tu Nombre: ")).lower()

path = 'Imagenes/' + nombreID

isExist = os.path.exists(path)

if isExist:
    print("Error: el Nombre ya está registrado")
    nombreID = str(input("Ingresa tu Nombre de nuevo: "))
else:
    os.makedirs(path)
    
while True:
    ret, frame = video.read()
    faces = cascada_cara.detectMultiScale(frame, 1.3, 5)
    
    for x, y, w , h in faces:
        count = count + 1
        nombre = './Imagenes/'+nombreID+'/'+str(count)+'.jpg'
        print("... Creando Imagenes ..." + nombre)
        
        cv2.imwrite(nombre, frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("WindowFrame", frame)
    k = cv2.waitKey(1)
        
    if count > 500:
        break
video.release()
cv2.destroyAllWindows()