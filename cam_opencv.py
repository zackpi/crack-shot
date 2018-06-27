import cv2
import numpy as np
from collections import deque

cam = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
haar_eye_cascade = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_eye.xml')

def avg_of_valid(queue):
    nvalid = 0
    rx,ry,rw,rh = 0,0,0,0
    for mem in queue:
        if mem:
            nvalid += 1
            rx += mem[0]
            ry += mem[1]
            rw += mem[2]
            rh += mem[3]
    if nvalid:
        return rx//nvalid, ry//nvalid, rw//nvalid, rh//nvalid

FACE_MEM_LEN = 5
face_mem = deque([None, ]*FACE_MEM_LEN)

EYE_MEM_LEN = 10
lefteye_mem = deque([None, ]*EYE_MEM_LEN)
righteye_mem = deque([None, ]*EYE_MEM_LEN)

while cv2.waitKey(1000//30) == -1:
    print(face_mem)
    
    ret, frame = cam.read()
    draw = frame.copy()

    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        
        
        if len(faces) == 0:
            face_mem.append(None)
            face_box = avg_of_valid(face_mem)
            if face_box:
                x,y,w,h = face_box
            else:
                face_mem.popleft()
                cv2.imshow("Full", frame)
                continue
                
            
        elif len(faces) > 1:
            # use the face closest to avg of prev faces
            face_box = avg_of_valid(face_mem)
            if face_box:
                px,py,pw,ph = face_box
                min_diff, min_i = 0, 0
                for i, (tx,ty,tw,th) in enumerate(faces):
                    diff = (px-tx)**2+(py-ty)**2+(pw-tw)**2+(ph-th)**2
                    if diff < min_diff:
                        min_diff = diff
                        min_i = i
            x,y,w,h = faces[i] 
            face_mem.append((x,y,w,h))   
        else:    
            x,y,w,h = faces[0]
            face_mem.append((x,y,w,h))
        face_mem.popleft()

        face = frame[y:y+h, x:x+w]        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 4)
        cv2.line(frame, (x,y+h//2), (x+w,y+h//2), (255,0,0), 2)
        cv2.line(frame, (x+w//2,y), (x+w//2,y+h), (255,0,0), 2)
        
        gray_face = gray_img[y:y+h, x:x+w]
        eyes = haar_eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1,
                                                minNeighbors=5)
        """        
        for (ex,ey,ew,eh) in eyes:
            if ey+eh//2 < h//2:
                cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 4)
                
                if ex+ew//2 < w//2: 
                    righteye_mem.popleft()
                    righteye_mem.append((ex, ey, ew, eh))
                    
                    rx,ry,rw,rh = avg_of_valid(righteye_mem)
                    
                    eye = face[ry:ry+rh, rx:rx+rw]
                    cv2.imshow("RightEye", eye)
                    
                else:
                    lefteye_mem.popleft()
                    lefteye_mem.append((ex, ey, ew, eh))
                    
                    lx,ly,lw,lh = avg_of_valid(lefteye_mem)
                    
                    eye = face[ly:ly+lh, lx:lx+lw]
                    cv2.imshow("LeftEye", eye)
        """
        cv2.imshow("Full", frame)
   
    else:
        break

