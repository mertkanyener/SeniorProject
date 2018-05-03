from sklearn.svm import SVC
from accuracy import Accuracy


class SVM:

    def __init__(self, kernel=None, C=None):
        self.kernel = kernel
        self.C = C

    def run_svm(self, X_train, X_test, y_train, y_test, C=1.0):

        svm_model = SVC(kernel='linear', C=C, random_state=1)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

        accuracy = Accuracy()
        accuracy = accuracy.calc_accuracy(y_pred, y_test)

        #print("y_pred : ", y_pred)
        print("SVM = C: ", C, "Accuracy : %", accuracy)




