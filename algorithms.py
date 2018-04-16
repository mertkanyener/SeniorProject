from vectorize import Vectorize
from sklearn.model_selection import train_test_split
from svm import SVM
from forest import RandomForest
from regression import LogRegression
from knn import Knn


dir1 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/female"
dir2 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_18-29_Neutral_bmp/male"
dir3 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/female"
dir4 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_30-49_Neutral_bmp/male"
dir5 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/female"
dir6 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_50-69_Neutral_bmp/male"
dir7 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/female"
dir8 = "/home/mertkanyener/Desktop/Uni/Senior project/dataset_cut/BW_age_70-94_Neutral_bmp/male"
dataset = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]


def preprocessing(dataset):
    vectorize = Vectorize()
    x_mean, y_mean = vectorize.get_avg_size(dataset)
    X, y_gender, y_age = vectorize.vectorize(dataset, x_mean, y_mean)
    #X_padded = vectorize.padding(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_gender, random_state=1, test_size=0.33, stratify=y_gender)
    return X_train, X_test, y_train, y_test


svm = SVM()
knn = Knn()
forest = RandomForest()
lr = LogRegression()
split_data = preprocessing(dataset)

svm.run_svm(split_data)
knn.run_knn(split_data)
forest.run_forest(split_data)
lr.run_lr(split_data)