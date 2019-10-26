####################################################
# Pragrammed By Satish Kumar					   #
# Download code from https://github.com/satish3922 #
# Bug Reporting @ hhtps://linked.com/in/satish3922 #
####################################################

# Importing required modules
import cv2
import os
from PIL import Image

# Getting datasets path where image stored
img_path = os.getcwd()+'/DataSets/'
img_ID = len(os.listdir(img_path))//100 + 1
face_cascade = cv2.CascadeClassifier('OpenCV-Python/HaarCascade/haarcascade_frontalface_default.xml')

# Define method to capture_from_webcam 100 pictures
def capture_from_webcam():
	sub_name = input('Subject Name : ')
	webcam = cv2.VideoCapture(0)
	count = 1
	while True:
		ret, frame = webcam.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		font = cv2.FONT_HERSHEY_SIMPLEX
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
		for (x,y,w,h) in faces:
			roi_gray = gray[y:y+h,x:x+w]
			cv2.imwrite(img_path+sub_name+'.'+str(img_ID)+'.'+str(count)+'.jpg',roi_gray)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
			cv2.putText(frame, sub_name, (x,y-4), font, 1, (0,0,255), 2, cv2.LINE_AA)
			count += 1
		cv2.imshow(sub_name,frame)
		cv2.waitKey(50)
		if count > 100:
			break

	webcam.release()
	cv2.destroyAllWindows()

capture_from_webcam()
