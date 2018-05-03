from sklearn.linear_model import LogisticRegression
from accuracy import Accuracy


class LogRegression:

    def __init__(self, C=None):
        self.C = C

    def run_lr(self, X_train, X_test, y_train, y_test, C=1.0):
        lr = LogisticRegression(C=C, random_state=1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = Accuracy()
        accuracy = accuracy.calc_accuracy(y_pred, y_test)

        #print("y_pred : ", y_pred)

        print("Logistic Regression = C: ", C, "Accuracy : %", accuracy)



