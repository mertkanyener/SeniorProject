from sklearn.neighbors import KNeighborsClassifier
from accuracy import Accuracy


class Knn:

    def __init__(self, neighbors=None):
        self.neighbors = neighbors

    def run_knn(self, X_train, X_test, y_train, y_test, neighbors=5):
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = Accuracy()
        accuracy = accuracy.calc_accuracy(y_pred, y_test)

        #print("y_pred : ", y_pred)

        print("KNN = k : ", neighbors, "Accuracy : %", accuracy)






