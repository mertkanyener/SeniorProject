import numpy as np
import cv2
import dlib
import os
import sys

from imutils import face_utils


def create_rect(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    face_rect = (x, y, w, h)
    return face_rect


def face_landmarks(image_name):
    try:
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        image = cv2.imread(image_name)

        rectangles = face_detector(image, 1)
        shape = 0
        for (i, rectangle) in enumerate(rectangles):

            shape = shape_predictor(image, rectangle)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = create_rect(rectangle)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if type(shape) != int:
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.imwrite(image_name, image)
        else:
            print(image_name + "-- not processed")
    except TypeError:
        print("there's no such image")
        raise
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


def landmarks_lib(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            image = os.path.join(subdir, file)
            print(file)
            face_landmarks(image)
    print("Images have been processed successfully.")


dir1 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]

for n in dataset:
    landmarks_lib(n)





