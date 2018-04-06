class Accuracy:

    def __init__(self, y_pred=None, y_test=None):
        self.y_pred = y_pred
        self.y_test = y_test

    def calc_accuracy(self, y_pred, y_test):
        correct_count = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                correct_count += 1
        result = (correct_count / len(y_test)) * 100
        return result




