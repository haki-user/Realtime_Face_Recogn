import cv2
import face_recognition
import os


file_name = os.path.join(os.path.dirname(__file__), 'images/m5.webp')
file_name2 = os.path.join(os.path.dirname(__file__), 'images/m3.jpg')


# encoding first image
# img = cv2.imread('messi1.jpeg')
img = cv2.imread(file_name)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

# encoding second image
img2 = cv2.imread(file_name2)
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

#comp
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)