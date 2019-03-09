import cv2
import numpy as py
from matplotlib import pyplot as plt 

#Loads haar classifier from local file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Loads video file, replace this path as necessary
cap = cv2.VideoCapture("resources/video/mari.mp4")

#Array containing coordinates
path = []

#This loop applies a grayscale filter to the video feed, 
#and applies the haar cascade to the resulting output.
#It then draws a rectangle around the forehead of the face,
#and appends the coordinates of that rectangle to the path array
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 9)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, int(y+h/3)), (255,0,0), 2)
        path.append((x,y))
       

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#Prints the coordinates
print (path)
cap.release()
#Ends the video session
cv2.destroyAllWindows()

