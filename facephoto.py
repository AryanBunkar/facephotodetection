# import numpy as np
# import cv2
# import os
# import csv
# import face_recognition
# from datetime import datetime

# img = cv2.imread(r"photo\groupphot.jpg")

# aryan_image = face_recognition.load_image_file(r"photo\aryan.jpg")
# aryan_encoding = face_recognition.face_encodings(aryan_image)[0]

# tanmay_image = face_recognition.load_image_file(r"photo\tanmay.jpg")
# tanmay_encoding = face_recognition.face_encodings(tanmay_image)[0]

# harshit_image = face_recognition.load_image_file(r"photo\harshit.jpg")
# harshit_encoding = face_recognition.face_encodings(harshit_image)[0]

# pranav_image = face_recognition.load_image_file(r"photo\pranav.jpg")
# pranav_encoding = face_recognition.face_encodings(pranav_image)[0]


# tanvi_image = face_recognition.load_image_file(r"photo\tanvi.jpg")
# tanvi_encoding  = face_recognition.face_encodings(tanvi_image)[0]

# vaibhavi_image = face_recognition.load_image_file(r"photo\vaibhavi.jpg")
# vaibhavi_encoding = face_recognition.face_encodings(vaibhavi_image)[0]

# mritun_image =face_recognition.load_image_file(r"photo\mritunjai.jpeg")
# mritun_encoding = face_recognition.face_encodings(mritun_image)[0]

# known_face_encodings = [
#     aryan_encoding,
#     harshit_encoding,
#     tanmay_encoding,
#     pranav_encoding,
#     tanvi_encoding,
#     vaibhavi_encoding,
#     mritun_encoding
# ]

# known_faces_names = [
#     "Aryan Bunkar",
#     "harshit",
#     "tanmay",
#     "pranav",
#     "tanvi",
#     "vaibhavi",
#     "mritunjai"
# ]


# students = known_faces_names.copy()


# face_location = []
# face_encodings = []
# face_names = []
# s = True


# now = datetime.now()
# current_date = now.strftime("%Y-%m-%d")

# f = open(current_date + ".csv","w+",newline="")
# lnwriter = csv.writer(f)

# # img = cv2.resize(img,(500,500))
# rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# face_location = face_recognition.face_locations(rgb_img)
# face_encodings = face_recognition.face_encodings(rgb_img,face_location)
# face_names = []

# for face_encoding in face_encodings:
#     matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
#     name = "Unknown"
#     face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
#     best_match_index = np.argmin(face_distance)
#     if matches[best_match_index]:
#         name = known_faces_names[best_match_index]

#     face_names.append(name)
#     if name in students:
#         students.remove(name)
#         print(name + " is present")
#         current_time = now.strftime("%H:%M:%S")
#         lnwriter.writerow([name, current_time])


# for (top,right,bottom,left),name in zip(face_location,face_names):
#     cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
#     cv2.putText(img,name,(left,bottom+20),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),2)
    
# cv2.imshow("face recognition",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import numpy as np
import cv2
import os
import csv
import face_recognition
from datetime import datetime

# Path where all student images are stored
IMAGE_DIR = "photo"

# Load known faces dynamically
known_face_encodings = []
known_faces_names = []

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # only add if face found
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]  # filename without extension
            known_faces_names.append(name)
        else:
            print(f"⚠️ No face found in {filename}")

students = known_faces_names.copy()

# Read group photo
img = cv2.imread(os.path.join("groupphot.jpg"))
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(rgb_img)
face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(current_date + ".csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in students:
            students.remove(name)
            print(f"{name} is present")
            lnwriter.writerow([name, now.strftime("%H:%M:%S")])

for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(img, name, (left, bottom + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
