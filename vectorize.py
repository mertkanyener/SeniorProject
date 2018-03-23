import numpy as np
import cv2
import os


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


def vectorize(dataset):
    #an empty list for every age group

    """
    age18_29_f = []
    age18_29_m = []
    age30_49_f = []
    age30_49_m = []
    age50_69_f = []
    age50_69_m = []
    age70_94_f = []
    age70_94_m = []
    """
    X = []
    y = []
    count = 0

    for group in dataset:
        count += 1
        for subdir, dirs, files in os.walk(group):
            for file in files:
                img_name = os.path.join(subdir, file)
                img = cv2.imread(img_name)
                img_v = np.matrix.flatten(img)  # vectorizing matrix

                if count == 1:
                    #age18_29_f.append(img_v)
                    X.append(img_v)
                    y.append(0)
                elif count == 2:
                    #age18_29_m.append(img_v)
                    X.append(img_v)
                    y.append(1)
                elif count == 3:
                    #age30_49_f.append(img_v)
                    X.append(img_v)
                    y.append(0)
                elif count == 4:
                    #age30_49_m.append(img_v)
                    X.append(img_v)
                    y.append(1)
                elif count == 5:
                    #age50_69_f.append(img_v)
                    X.append(img_v)
                    y.append(0)
                elif count == 6:
                    X.append(img_v)
                    y.append(1)
                    #age50_69_m.append(img_v)
                elif count == 7:
                    X.append(img_v)
                    y.append(0)
                    #age70_94_f.append(img_v)
                elif count == 8:
                    X.append(img_v)
                    y.append(1)
                    #age70_94_m.append(img_v)

    #arr = np.array([age18_29_f, age18_29_m, age30_49_f, age30_49_m, age50_69_f, age50_69_m, age70_94_f, age70_94_m])

    #X = arr.reshape(4, 2)  #result matrix 4 rows for 4 age groups, 2 cols for genders
    return X, y


X, y = vectorize(dataset)

#print(X[:, 0][0][0])
#print(X)
#print(y)





