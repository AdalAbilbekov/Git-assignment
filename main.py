import numpy as np
import cv2
import face_recognition as face_rec

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# img declration
adal = face_rec.load_image_file('sample_images/adal.jpg')
adal = cv2.cvtColor(adal, cv2.COLOR_BGR2RGB)
adal = resize(adal, 0.50)
adal_test = face_rec.load_image_file('sample_images/adal_test.jpg')
adal_test = cv2.cvtColor(adal_test, cv2.COLOR_BGR2RGB)
adal_test = resize(adal_test, 0.50)

# face location
faceLocation_adal = face_rec.face_locations(adal)[0]
encode_adal = face_rec.face_encodings(adal)[0]
cv2.rectangle(adal, (faceLocation_adal[3], faceLocation_adal[0]), (faceLocation_adal[1], faceLocation_adal[2]), (255, 0, 255), 3)

faceLocation_adal_test = face_rec.face_locations(adal_test)[0]
encode_adal_test = face_rec.face_encodings(adal_test)[0]
cv2.rectangle(adal_test, (faceLocation_adal_test[3], faceLocation_adal_test[0]), (faceLocation_adal_test[1], faceLocation_adal_test[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_adal], encode_adal_test)
print(results)
cv2.putText(adal_test, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('main_jpg', adal)
cv2.imshow('test_jpg', adal_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
