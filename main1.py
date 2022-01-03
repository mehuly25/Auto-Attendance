import cv2
import face_recognition
import numpy

#STEP 1
imgElon=face_recognition.load_image_file('elonmusk_trainimg.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElonTest=face_recognition.load_image_file('mehulytest.jpg')
imgElonTest=cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

#STEP 2
faceloc = face_recognition.face_locations(imgElon)[0]  #(39, 163, 101, 100)
encodeElon = face_recognition.face_encodings(imgElon)[0]
#print(faceloc)
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,255,0),2)


facelocTest = face_recognition.face_locations(imgElonTest)[0]  #(39, 163, 101, 100)
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,255,0),2)

#STEP 3
result=face_recognition.compare_faces([encodeElon],encodeElonTest)
facedist=face_recognition.face_distance([encodeElon],encodeElonTest)
print(result,facedist)
cv2.putText(imgElonTest,f'{result} {round(facedist[0],2)}',(20,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),2)


cv2.imshow('ElonTest', imgElonTest)
cv2.imshow('Elon', imgElon)
cv2.waitKey(0)
