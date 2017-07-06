import cv2
import numpy as np
import dlib

image_path = "testing_photo.JPG"

# Create the haar cascade
faceCascade =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create the landmark predictor
predictor =  dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread(image_path)
#resizing according to the image, done manually for instance
image = cv2.resize(image, (1000,700))

# convert the image to grayscale
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Converting the OpenCV rectangle coordinates to Dlib rectangle
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

    detected_landmarks = predictor(image, dlib_rect).parts()

    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

    # print landmarks
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])

        # annotate the positions
        # cv2.putText(img=image_copy, text=str(idx),org=pos,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.4,
        #             color=(0, 0, 255))

        # draw points on the landmark positions
        cv2.circle(image, pos, 2, color=(0, 0, 255),thickness=-5)


# cv2.imshow("Faces found", image)
cv2.imshow("Landmarks found", image)
#saving the picture
# cv2.imwrite("testing_photo_output.JPG",image)
cv2.waitKey(0)

