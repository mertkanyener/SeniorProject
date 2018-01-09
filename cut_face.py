import cv2
import dlib
import numpy as np
import facial_landmark as fl

from imutils import face_utils


# This function cuts the face from images

def cut_face(image_name):


    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_name)

    rectangles = face_detector(image, 1)
    for (i, rectangle) in enumerate(rectangles):
        shape = shape_predictor(image, rectangle)
        shape = face_utils.shape_to_np(shape)
        x = shape[1]
        y = shape[25]
        w = shape[16]
        h = shape[9]
        cv2.rectangle(image, (x[0], y[1] - 20), (w[0], h[1] + 10), (0, 255,0), 2)
    face = image[(y[1]-20):(h[1]+10), x[0]:w[0]]

    cv2.imshow('img',face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



cut_face('img3.bmp')

