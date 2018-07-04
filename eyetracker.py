import sys
import cv2
import numpy as np
from collections import deque

class EyeTracker:
    FACE_CASC = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
    EYE_CASC = cv2.CascadeClassifier('/usr/local/bin/opencv/data/haarcascades/haarcascade_eye.xml')
    
    FACE_MEM_LEN = 5
    EYE_MEM_LEN = 10
    
    def __init__(self, source=0):
        self.cam = cv2.VideoCapture(source)
        self.img = None
        self.draw = None
        
        self.face_mem = deque(maxlen=EyeTracker.FACE_MEM_LEN)
        self.reye_mem = deque(maxlen=EyeTracker.EYE_MEM_LEN)
        self.leye_mem = deque(maxlen=EyeTracker.EYE_MEM_LEN)
        
    self.assim = lambda i,v,l: v/(l-i)                  # assimilation function
        self.dist = lambda x,y,h,k: (x-h)**2+(y-k)**2   # distance function
        
    def nextFrame(self):
        ret, frame = self.cam.read()
        if ret:
            self.img = frame
            self.draw = self.img.copy()
            return True
        return False
    
    def assimilate(self, q):
        l = len(q)
        if l:
            norm = sum([self.assim(i,1,l) for i in range(len(q))])
            
            rx,ry,rw,rh = 0,0,0,0
            for i, (x,y,w,h) in enumerate(q):
                rx += self.assim(i,x,l)
                ry += self.assim(i,y,l)
                rw += self.assim(i,w,l)
                rh += self.assim(i,h,l)
            
            return int(rx/norm), int(ry/norm), int(rw/norm), int(rh/norm)
    
    def detectFeature(self, casc, mem):
        
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        boxes = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if not len(boxes):
            if mem:
                mem.popleft()
            return self.assimilate(mem)
        
        x,y,w,h = boxes[0]    
        if len(boxes) > 1:
            # use the face closest to avg of prev faces
            box = self.assimilate(mem)
            if box:
                bx,by,bw,bh = box
                mindiff = lambda _x,_y,_w,_h: -self.dist(_x,_y,bx,by) - self.dist(_w,_h,bw,bh)
                x,y,w,h = max(boxes, key=mindiff)
            
        mem.append((x,y,w,h))
        return x,y,w,h

if __name__=="__main__":
    
    e = EyeTracker()
    while cv2.waitKey(1000//30) == -1:
        e.nextFrame()
        face = e.detectFeature(e.FACE_CASC, e.face_mem)
        
        frame = e.draw
        if face:
            x,y,w,h = face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 4)
            cv2.line(frame, (x,y+h//2), (x+w,y+h//2), (255,0,0), 2)
            cv2.line(frame, (x+w//2,y), (x+w//2,y+h), (255,0,0), 2)
        cv2.imshow("", frame)
    
    
    
    
    
    
    
    
    
    
    
    
    

        
    
    
