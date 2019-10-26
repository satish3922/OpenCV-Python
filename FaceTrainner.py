####################################################
# Pragrammed By Satish Kumar					   #
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
img_path = os.getcwd()+'/DataSets/'
img_list = os.listdir(img_path)
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define method to get_image_with_ID
def get_image_with_ID(img_list):
    IDs = []
    Images = []
    image_ID_dict = {0:'Unknown'}
    font = cv2.FONT_HERSHEY_SIMPLEX # Font of Text

    for image in img_list:
        img = Image.open(os.path.join(img_path+image)).convert('L')
        img_array = np.array(img,'uint8')
        Ids = int(image.split('.')[1])
        image_ID_dict[Ids] = image.split('.')[0]
        faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.2, minNeighbors=5)
        for x,y,w,h in faces:
            Images.append(img_array[y:y+h,x:x+w])
            IDs.append(Ids)
            # Create a rectangle around the faces
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0,255,0), 4)

            # Putting Text Outside the rectangle
            cv2.putText(img_array, image.split('.')[0], (x,y+20), font, 1, (0,0,255), 2, cv2.LINE_AA)

        # Showing the image
        cv2.imshow('Training',img_array)
        cv2.waitKey(5)
    return Images,np.array(IDs),image_ID_dict

Images,IDs,image_ID_dict = get_image_with_ID(img_list)
# print(type(IDs),type(Images),len(IDs),len(Images))
with open("Pickles/image_ID_dict.pickle","wb") as f:
    pickle.dump(image_ID_dict,f)
recognizer.train(Images,IDs)
recognizer.save('Recognizer/LBPHFaceRecognizer.yml')
cv2.destroyAllWindows()
print('LBPHFaceRecognizer trained successfully!')
