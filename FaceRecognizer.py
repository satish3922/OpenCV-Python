####################################################
# Pragrammed By Satish Kumar                       #
# Download code from https://github.com/satish3922 #
# Bug Reporting @ hhtps://linked.com/in/satish3922 #
####################################################

# Importing required modules
import cv2
import os
from PIL import Image
import numpy as np
import pickle

# Getting datasets path where the image is stored
img_path = os.getcwd()
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')

# Read the LBPHFaceRecognizer models
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Recognizer/LBPHFaceRecognizer.yml')

# Load the image_ID_dict
image_ID_dict = {}
with open('Pickles/image_ID_dict.pickle', 'rb') as f:
    image_ID_dict = pickle.load(f)

# Load the webcam
webcam = cv2.VideoCapture(0)

# Define a method to Recognize the pictures
while True:

    ret,frame = webcam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX # Font of Text
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for x,y,w,h in faces:
        roi_gray = gray[y:y+h,x:x+w]
        ID,Confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(ID,Confidence)
        name = ''
        # Check if Exists or Not
        if Confidence < 85:
            name = image_ID_dict[ID]
        else:
            name = 'Unknown'

        # Create a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 4)

        # Putting Text Outside the rectangle
        cv2.putText(frame, name, (x,y-6), font, 1, (255,255,255), 2, cv2.LINE_AA)

    # Showing the image
    cv2.imshow('Recognizing',frame)

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# Stop the Camera
webcam.release()

# Close all windows
cv2.destroyAllWindows()
