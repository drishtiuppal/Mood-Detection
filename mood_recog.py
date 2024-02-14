import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import dlib

data=np.load("C:/Users/Dell/OneDrive/Desktop/vs/python/face_mood.npy")



x=data[:,1:].astype(int)
y=data[:,0]

model=KNeighborsClassifier()
model.fit(x,y)

import cv2 as cv



cap=cv.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:/Users/Dell/OneDrive/Desktop/vs/python/shape_predictor_68_face_landmarks.dat") 


while True:
    ret,frame=cap.read() 
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=detector(frame)
    for face in faces:
        landmark=predictor(gray,face)
        # print(landmark.parts())
        # nose=landmark.parts()[27]
        # print(nose.x,nose.y)
        lip_up=landmark.parts()[62].y
        lip_down=landmark.parts()[66].y
        # print("down",lip_down)
        # print(lip_up)
        # if lip_down-lip_up>11:
        #     print("mouth open")
        # else:
        #     print("mouth closed")    
            
            
        expression=np.array([[point.x-face.left(),point.y-face.top()] for point in landmark.parts()[17: ]])  
        # print(expression.flatten())  
        print(model.predict([expression.flatten()]))
            
            
        for points in landmark.parts()[17: ]:
            cv.circle(frame,(points.x,points.y),2,(0,0,255))
        
    # print(faces)
    if ret:
        
        cv.imshow("window", frame)
        
    key=cv.waitKey(1)     
    
    if key == ord("q"):
        break
    
cap.release()
cv.destroyAllWindows()