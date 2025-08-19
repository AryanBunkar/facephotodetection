import numpy as np
import cv2
import os
import csv
import face_recognition
from datetime import datetime

img = cv2.imread(r"C:\Users\HP\Pictures\aryann\groupphot.jpg")

aryan_image = face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\aryan.jpg")
aryan_encoding = face_recognition.face_encodings(aryan_image)[0]

tanmay_image = face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\tanmay.jpg")
tanmay_encoding = face_recognition.face_encodings(tanmay_image)[0]

harshit_image = face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\harshit.jpg")
harshit_encoding = face_recognition.face_encodings(harshit_image)[0]

pranav_image = face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\pranav.jpg")
pranav_encoding = face_recognition.face_encodings(pranav_image)[0]


tanvi_image = face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\tanvi.jpg")
tanvi_encoding  = face_recognition.face_encodings(tanvi_image)[0]

vaibhavi_image = face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\vaibhavi.jpg")
vaibhavi_encoding = face_recognition.face_encodings(vaibhavi_image)[0]

mritun_image =face_recognition.load_image_file(r"C:\Users\HP\Desktop\goody\opencv\facedetection attendence\photo\mritunjai.jpeg")
mritun_encoding = face_recognition.face_encodings(mritun_image)[0]

known_face_encodings = [
    aryan_encoding,
    harshit_encoding,
    tanmay_encoding,
    pranav_encoding,
    tanvi_encoding,
    vaibhavi_encoding,
    mritun_encoding
]

known_faces_names = [
    "Aryan Bunkar",
    "harshit",
    "tanmay",
    "pranav",
    "tanvi",
    "vaibhavi",
    "mritunjai"
]


students = known_faces_names.copy()


face_location = []
face_encodings = []
face_names = []
s = True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + ".csv","w+",newline="")
lnwriter = csv.writer(f)

# img = cv2.resize(img,(500,500))
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

face_location = face_recognition.face_locations(rgb_img)
face_encodings = face_recognition.face_encodings(rgb_img,face_location)
face_names = []

for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
    name = "Unknown"
    face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
    best_match_index = np.argmin(face_distance)
    if matches[best_match_index]:
        name = known_faces_names[best_match_index]

    face_names.append(name)
    if name in students:
        students.remove(name)
        print(name + " is present")
        current_time = now.strftime("%H:%M:%S")
        lnwriter.writerow([name, current_time])


for (top,right,bottom,left),name in zip(face_location,face_names):
    cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
    cv2.putText(img,name,(left,bottom+20),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),2)
    
cv2.imshow("face recognition",img)
cv2.waitKey(0)
cv2.destroyAllWindows()