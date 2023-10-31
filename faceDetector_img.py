import cv2

# loading some pre-trained data on face frontals using opencv (haar cascade alg)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an img to face detect
img = cv2.imread('RDJ.jpg')

# must convert to grayscale for less factors
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# actualy detect the faces
face_coords = trained_face_data.detectMultiScale(grayscale_img)
# detectMultiScale detects the faces no matter the scale of the face

print(face_coords) # now that the coords have been obtained we must draw the square around the face
# [upper left (x,y) and bottom right (x,y)]

# draw
(x,y,w,h) = face_coords[0]
for (x,y,w,h) in face_coords:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0) ,1)

# test
cv2.imshow('Face Detector', img)
cv2.waitKey()



# checks if the code finishes
print("Code Done! ;)")