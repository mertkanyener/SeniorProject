import numpy as np
import cv2
import os


class Vectorize:

    def __init__(self, dataset=None):
        self.dataset = dataset

    def vectorize(self, dataset):
        X = []
        y_gender = []
        y_age = []
        count = 0

        for group in dataset:
            count += 1
            for subdir, dirs, files in os.walk(group):
                for file in files:
                    img_name = os.path.join(subdir, file)
                    img = cv2.imread(img_name)
                    img_v = np.matrix.flatten(img)  # vectorizing matrix
                    X.append(img_v)
                    if count == 1:
                        y_age.append(1)
                        y_gender.append(0)
                    elif count == 2:
                        y_age.append(1)
                        y_gender.append(1)
                    elif count == 3:
                        y_age.append(2)
                        y_gender.append(0)
                    elif count == 4:
                        y_age.append(2)
                        y_gender.append(1)
                    elif count == 5:
                        y_age.append(3)
                        y_gender.append(0)
                    elif count == 6:
                        y_age.append(3)
                        y_gender.append(1)
                    elif count == 7:
                        y_age.append(4)
                        y_gender.append(0)
                    elif count == 8:
                        y_age.append(4)
                        y_gender.append(1)

        return X, y_gender, y_age
        # arr = np.array([age18_29_f, age18_29_m, age30_49_f, age30_49_m, age50_69_f, age50_69_m, age70_94_f, age70_94_m])

        # X = arr.reshape(4, 2)  #result matrix 4 rows for 4 age groups, 2 cols for genders



    def padding(self, X):
        X_padded = []
        size_list = []
        for vec in X:
            size_list.append(vec.shape[0])
        max_size = max(size_list)
        for vec in X:
            difference = max_size - vec.shape[0]
            vec = np.pad(vec, (0, max_size - vec.shape[0]), 'mean')
            # vec = np.pad(vec, (int(difference/2), int(difference/2)), 'constant', constant_values=0)
            X_padded.append(vec)
        return X_padded




"""
# dataset directories needs to be edited according to pc path
dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]
"""







#X, y = vectorize(dataset)

#print(X[:, 0][0][0])
#print(X)
#print(y)





