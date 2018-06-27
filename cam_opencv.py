import cv2
import numpy as np
from collections import deque

cam = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
haar_eye_cascade = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_eye.xml')

FACE_MEM_LEN = 5
face_mem = deque([(0, 0, 0, 0), ]*FACE_MEM_LEN)

EYE_MEM_LEN = 10
lefteye_mem = deque([(0, 0, 0, 0), ]*EYE_MEM_LEN)
righteye_mem = deque([(0, 0, 0, 0), ]*EYE_MEM_LEN)

while cv2.waitKey(1000//30) == -1:
    print(face_mem)
    
    ret, frame = cam.read()
    draw = frame.copy()

    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        
        
        if len(faces) != 1:
            fx = sum([s[0] for s in face_mem]) // FACE_MEM_LEN
            fy = sum([s[1] for s in face_mem]) // FACE_MEM_LEN
            fw = sum([s[2] for s in face_mem]) // FACE_MEM_LEN
            fh = sum([s[3] for s in face_mem]) // FACE_MEM_LEN
        
            if len(faces) == 0:
                # use average of previous faces as new face outline
                x,y,w,h = fx,fy,fw,fh
            elif len(faces) > 1:
                # use the face closest to avg of prev faces
                min_diff, min_i = 0, 0
                for i, (tx,ty,tw,th) in enumerate(faces):
                    diff = (fx-tx)**2+(fy-ty)**2+(fw-tw)**2+(fh-th)**2
                    if diff < min_diff:
                        min_diff = diff
                        min_i = i
                x,y,w,h = faces[i]    
        else:    
            x,y,w,h = faces[0]
        
        face_mem.popleft()
        face_mem.append((x,y,w,h))
                
        face = frame[y:y+h, x:x+w]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 4)
        cv2.line(frame, (x,y+h//2), (x+w,y+h//2), (255,0,0), 2)
        cv2.line(frame, (x+w//2,y), (x+w//2,y+h), (255,0,0), 2)
        
        gray_face = gray_img[y:y+h, x:x+w]
        eyes = haar_eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1,
                                                minNeighbors=5)
        
        for (ex,ey,ew,eh) in eyes:
            if ey+eh//2 < h//2:
                cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 4)
                
                if ex+ew//2 < w//2: 
                    righteye_mem.popleft()
                    righteye_mem.append((ex, ey, ew, eh))
                    
                    sx = sum([s[0] for s in righteye_mem]) // EYE_MEM_LEN
                    sy = sum([s[1] for s in righteye_mem]) // EYE_MEM_LEN
                    sw = sum([s[2] for s in righteye_mem]) // EYE_MEM_LEN
                    sh = sum([s[3] for s in righteye_mem]) // EYE_MEM_LEN
                    
                    eye = face[sy:sy+sh, sx:sx+sw]
                    cv2.imshow("RightEye", eye)
                    
                else:
                    lefteye_mem.popleft()
                    lefteye_mem.append((ex, ey, ew, eh))
                    
                    sx = sum([s[0] for s in lefteye_mem]) // EYE_MEM_LEN
                    sy = sum([s[1] for s in lefteye_mem]) // EYE_MEM_LEN
                    sw = sum([s[2] for s in lefteye_mem]) // EYE_MEM_LEN
                    sh = sum([s[3] for s in lefteye_mem]) // EYE_MEM_LEN
                    
                    eye = face[sy:sy+sh, sx:sx+sw]
                    cv2.imshow("LeftEye", eye)
        cv2.imshow("Full", frame)
   
    else:
        break

