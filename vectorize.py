import numpy as np
import cv2
import os



dir1 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/senior_project/dataset_lm/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]


def vectorize(dataset):
    age18_29_f = []
    age18_29_m = []
    age30_49_f = []
    age30_49_m = []
    age50_69_f = []
    age50_69_m = []
    age70_94_f = []
    age70_94_m = []
    count = 0

    for group in dataset:
        count += 1
        for subdir, dirs, files in os.walk(group):
            for file in files:
                img_name = os.path.join(subdir, file)
                img = cv2.imread(img_name)
                img_v = np.matrix.flatten(img)
                if count == 1:
                    age18_29_f.append(img_v)
                elif count == 2:
                    age18_29_m.append(img_v)
                elif count == 3:
                    age30_49_f.append(img_v)
                elif count == 4:
                    age30_49_m.append(img_v)
                elif count == 5:
                    age50_69_f.append(img_v)
                elif count == 6:
                    age50_69_m.append(img_v)
                elif count == 7:
                    age70_94_f.append(img_v)
                elif count == 8:
                    age70_94_m.append(img_v)

    arr = np.array([age18_29_f, age18_29_m, age30_49_f, age30_49_m, age50_69_f, age50_69_m, age70_94_f, age70_94_m])
    m = arr.reshape(4, 2) #result matrix 4 rows for 4 age groups, 2 cols for genders
    return m


m = vectorize(dataset)
print(m)




