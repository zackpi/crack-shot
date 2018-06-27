import cv2
import numpy as np

cam = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

while cv2.waitKey(1000//30) == -1:
    
    ret, frame = cam.read()

    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces):
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (100, 100, 100), 1)
            cv2.imshow("Face", frame)
        else:
            print("No face found.")
    else:
        break

