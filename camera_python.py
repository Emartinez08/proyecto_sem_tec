import cv2
import time
import argparse


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Camera visualization')

    ### Positional arguments
    parser.add_argument('-i', '--cameraSource', default=0, help="Introduce number or camera path, default is 0 (default cam)")

    
    args = vars(parser.parse_args())

    # clasificador// manda a llamar uno de los clasificadores predefinidos de openCv
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    cap = cv2.VideoCapture(args["cameraSource"]) #0 local o primary camera
    while cap.isOpened():
        
        #BGR image feed from camera
        success,img = cap.read()
        # deteccion de rostro// usa el clasificador para detectar rostros
        faces = faceClassif.detectMultiScale(img, 1.3,5)  # si aumentan mucho los nueros detecata pocos rostro y viceversa
        #Agregar un rectangulo a la imagen
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # agrega un rectangulo a las caras detectadas

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


    print('Script took %f seconds.' % (time.time() - script_start_time))



