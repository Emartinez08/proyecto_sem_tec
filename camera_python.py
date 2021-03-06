import cv2
import time
import argparse
import imutils


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Camera visualization')

    ### Positional arguments
    parser.add_argument('-i', '--cameraSource', default=0, help="Introduce number or camera path, default is 0 (default cam)")

    
    args = vars(parser.parse_args())

    # clasificador// manda a llamar uno de los clasificadores predefinidos de openCv
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # casco incrustable
    casco = cv2.imread("casco.png", cv2.IMREAD_UNCHANGED)



    cap = cv2.VideoCapture(args["cameraSource"]) #0 local o primary camera
    sentimiento = int(input("(1)¿¡Arriba México!? | (2)no "))
    while cap.isOpened():
        
        #BGR image feed from camera
        success,img = cap.read()
        # deteccion de rostro// usa el clasificador para detectar rostros
        faces = faceClassif.detectMultiScale(img, 1.3,5)  # si aumentan mucho los nueros detecata pocos rostro y viceversa
        #Agregar un rectangulo a la imagen
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # agrega un rectangulo a las caras detectadas

            # redimencionar imagen
            imagen_adaptada = imutils.resize(casco, width=w)
            filas_casco = imagen_adaptada.shape[0]
            col_casco = w

            #ajustar a la frente, disminuye una porcion de la imagen para ponerlo sobre la frente
            porcion = filas_casco // 4
            if y - filas_casco + porcion >= 0:
                img[y - filas_casco + porcion:y + porcion, x:x + w] = imagen_adaptada[:, :, 0:3]


        if not success:
            break
        if img is None:
            break


        #filtro gris
        img_gris=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        if sentimiento==1:
            cv2.imshow("Output", img)
        else:
            cv2.imshow("Output", img_gris)

        k = cv2.waitKey(10)
        if k==27:
            break


    cap.release()
    cv2.destroyAllWindows()


    print('Script took %f seconds.' % (time.time() - script_start_time))



