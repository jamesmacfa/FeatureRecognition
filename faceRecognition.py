import cv2
import numpy as py
from matplotlib import pyplot as plt 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("resources/video/mari.mp4")

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 9)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, int(y+h/3)), (255,0,0), 2)
       

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cap.destroyAllWindows()

