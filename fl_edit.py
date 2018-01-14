import cv2
import dlib
import numpy as np
import facial_landmark as fl

from imutils import face_utils




def facial_landmark_edit(image_name):


    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_name)

    rectangles = face_detector(image, 1)  # localizing the face area in the image
    for (i, rectangle) in enumerate(rectangles):
        shape = shape_predictor(image, rectangle)
        shape = face_utils.shape_to_np(shape)
        x = shape[1]
        y = shape[25]
        w = shape[16]
        h = shape[9]
        cv2.rectangle(image, (x[0], y[1]-10), (w[0], h[1]), (0,255,0), 2)  # put the interesting area of face in a rectangle



    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





facial_landmark_edit('')

