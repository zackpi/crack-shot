import sys
import cv2
import numpy as np
from collections import deque

sys.stderr = object     # don't print errors to command line

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
            min_diff, min_i = sys.maxsize, 0
            if face_box:
                px,py,pw,ph = face_box
                for i, (tx,ty,tw,th) in enumerate(faces):
                    diff = (px-tx)**2+(py-ty)**2+(px+pw-tx-tw)**2+(py+ph-ty-th)**2
                    if diff < min_diff:
                        min_diff = diff
                        min_i = i
            x,y,w,h = faces[min_i] 
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
        
        right_eyes = [(ex,ey,ew,eh) for (ex,ey,ew,eh) in eyes if ey+eh//2 < h//2 
                                                            and ex+ew//2 < w//2]
        if len(right_eyes) == 0:
            righteye_mem.append(None)
            reye_box = avg_of_valid(righteye_mem)
            if reye_box:
                ex,ey,ew,eh = reye_box
            else:
                righteye_mem.popleft()
                continue
        elif len(right_eyes) > 1:
            # use the face closest to avg of prev faces
            reye_box = avg_of_valid(righteye_mem)
            
            min_diff, min_i = sys.maxsize, 0
            if reye_box:
                px,py,pw,ph = reye_box
                for i, (tx,ty,tw,th) in enumerate(right_eyes):
                    diff = (px-tx)**2+(py-ty)**2+(px+pw-tx-tw)**2+(py+ph-ty-th)**2
                    if diff < min_diff:
                        min_diff = diff
                        min_i = i
            ex,ey,ew,eh = right_eyes[min_i] 
            righteye_mem.append((ex,ey,ew,eh))
        else:    
            ex,ey,ew,eh = right_eyes[0]
            righteye_mem.append((ex,ey,ew,eh))
        cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,0,255), 2)
        cv2.line(frame, (x+ex,y+ey+eh//2), (x+ex+ew,y+ey+eh//2), (0,0,255), 1)
        cv2.line(frame, (x+ex+ew//2,y+ey), (x+ex+ew//2,y+ey+eh), (0,0,255), 1)
        
        left_eyes = [(ex,ey,ew,eh) for (ex,ey,ew,eh) in eyes if ey+eh//2 < h//2 
                                                            and ex+ew//2 >= w//2]
        if len(left_eyes) == 0:
            lefteye_mem.append(None)
            leye_box = avg_of_valid(lefteye_mem)
            if leye_box:
                ex,ey,ew,eh = leye_box
            else:
                lefteye_mem.popleft()
                continue
        elif len(left_eyes) > 1:
            # use the face closest to avg of prev faces
            leye_box = avg_of_valid(lefteye_mem)
            
            min_diff, min_i = sys.maxsize, 0
            if leye_box:
                px,py,pw,ph = leye_box
                for i, (tx,ty,tw,th) in enumerate(left_eyes):
                    diff = (px-tx)**2+(py-ty)**2+(px+pw-tx-tw)**2+(py+ph-ty-th)**2
                    if diff < min_diff:
                        min_diff = diff
                        min_i = i
            ex,ey,ew,eh = left_eyes[min_i] 
            lefteye_mem.append((ex,ey,ew,eh))
        else:    
            ex,ey,ew,eh = left_eyes[0]
            lefteye_mem.append((ex,ey,ew,eh))   
        cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,0,255), 2)
        cv2.line(frame, (x+ex,y+ey+eh//2), (x+ex+ew,y+ey+eh//2), (0,0,255), 1)
        cv2.line(frame, (x+ex+ew//2,y+ey), (x+ex+ew//2,y+ey+eh), (0,0,255), 1)
        
        cv2.imshow("Full", frame)
   
    else:
        break

