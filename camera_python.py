import cv2
import time
import argparse
import imutils

# clasificador// manda a llamar uno de los clasificadores predefinidos de openCv
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# casco incrustable
imagen = cv2.imread("casco3.png", cv2.IMREAD_UNCHANGED)


cap = cv2.VideoCapture(0) 
while cap.isOpened():
    
    #BGR image feed from camera
    success,img = cap.read()
    # deteccion de rostro// usa el clasificador para detectar rostros
    faces = faceClassif.detectMultiScale(img, 1.3,5)  # si aumentan mucho los nueros detecata pocos rostro y viceversa
    #Agregar un rectangulo a la imagen
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(245,232,50),4) # agrega un rectangulo a las caras detectadas

        # redimencionar imagen
        imagen_adaptada = imutils.resize(imagen, width=w)
        filas_imagen = imagen_adaptada.shape[0]
        col_imagen = w
        if y - filas_imagen >= 0:
            img[y - filas_imagen:y, x:x + w] = imagen_adaptada[:, :, 0:3]

    if not success:
        break
    if img is None:
        break

    
    cv2.imshow("Output", img)

    k = cv2.waitKey(10)
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()





