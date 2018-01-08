import cv2
import dlib
import numpy as np
import facial_landmark as fl

from imutils import face_utils




def facial_landmark_edit(image_name):


    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_name)

    rectangles = face_detector(image, 1)
    for (i, rectangle) in enumerate(rectangles):
        shape = shape_predictor(image, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = fl.create_rect(rectangle)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count = 0
        for (x, y) in shape:
            count += 1
            if count == 49:
                mx = (x, y)
            elif count == 52:
                my = (x, y)
            elif count == 58:
                mh = (x, y)
            elif count == 55:
                mw = (x, y)
        cv2.rectangle(image, (mx[0]-10, my[1]-10), (mw[0]+10, mh[1]+10), (0, 255, 0), 2)

    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



facial_landmark_edit('image2.bmp')

