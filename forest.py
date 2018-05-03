from sklearn.ensemble import RandomForestClassifier
from accuracy import Accuracy


class RandomForest:

    def __init__(self, n_estimators=None, criteron=None):
        self.n_estimators = n_estimators
        self.criteron = criteron

    def run_forest(self, X_train, X_test, y_train, y_test, n_estimators=128, criterion='gini'):
        accuracy = Accuracy()
        forest = RandomForestClassifier(criterion=criterion,
                                        n_estimators=n_estimators,
                                        random_state=1,
                                        n_jobs=2)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        accuracy = accuracy.calc_accuracy(y_pred, y_test)

        # print("y_pred : ", y_pred)

        print("Random Forest = Tree number : ", n_estimators, "Criterion : ", criterion, " Accuracy : %", accuracy)







