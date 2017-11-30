import numpy as np
import cv2
import dlib
import imutils


from imutils import face_utils

def create_rect(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    face_rect = (x, y, w, h)
    return face_rect

def face_landmarks(image_name):

    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_name)
    #image = imutils.resize(image, width=500)
    rectangles = face_detector(image, 1)
    for (i, rectangle) in enumerate(rectangles):

        shape = shape_predictor(image, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = create_rect(rectangle)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

face_landmarks('image1.bmp')
face_landmarks('image2.bmp')
face_landmarks('image3.bmp')
face_landmarks('image4.bmp')

def func(a):
    return a * 2
    
