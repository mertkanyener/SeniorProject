import numpy as np

from vectorize import Vectorize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from svm import SVM
from forest import RandomForest
from regression import LogRegression
from knn import Knn



dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/Dataset/BW_age_70-94_Neutral_bmp/male"

dir1_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/female"
dir2_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/male"
dir3_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/female"
dir4_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/male"
dir5_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/female"
dir6_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/male"
dir7_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/female"
dir8_g = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/male"
dataset_age = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]
dataset_gender = [dir1_g, dir2_g, dir3_g, dir4_g, dir5_g, dir6_g, dir7_g, dir8_g]

def preprocessing(dataset, option):
    vectorize = Vectorize()
    x_mean, y_mean = vectorize.get_avg_size(dataset)
    X, X_age, y_gender, y_age = vectorize.vectorize(dataset, x_mean, y_mean)

    #X_padded = vectorize.padding(X)
    if option == 'gender':
        X_train, X_test, y_train, y_test = train_test_split(X, y_gender, random_state=1,
                                                            test_size=0.2, stratify=y_gender)
        return X_train, X_test, y_train, y_test
    elif option == 'age':
        X_train, X_test, y_train, y_test = train_test_split(X_age, y_age, random_state=1, test_size=0.2,
                                                            stratify=y_age)
        return X_train, X_test, y_train, y_test
    else:
        print('Unexpected option can be only "gender" or "age" ')
        return 0





svm = SVM()
knn = Knn()
forest = RandomForest()
lr = LogRegression()
X_train, X_test, y_train, y_test = preprocessing(dataset_age, 'age')




svm.run_svm(X_train, X_test, y_train, y_test)
knn.run_knn(X_train, X_test, y_train, y_test)
forest.run_forest(X_train, X_test, y_train, y_test)
lr.run_lr(X_train, X_test, y_train, y_test)

"""
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores = []

X_train = np.asanyarray(X_train)
print(type(X_train))

for k, (train, test) in enumerate(kfold):
    print("Train:",train)
    print(test)
    lr = LogisticRegression()
    lr.fit(X_train[train], y_train[train])
    score = lr.score(X_train[test], y_train[test])
    scores.append(score)


print('Log reg kfold Accuracy: %', np.mean(scores))
"""


