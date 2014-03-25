import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
import os


#def evaluation(size, log_C, log_gamma):
#def evaluation(size, log_C):
def evaluation(log_C):
    #X_train, y_train = load_svmlight_file("/home/kleinaa/data/mnist/mnist")
    #X_test, y_test = load_svmlight_file("/home/kleinaa/data/mnist/mnist.t")

    X, y = load_svmlight_file(os.path.dirname(os.path.abspath(__file__)) + "/data/mnist")
    X_train = X[0:40000]
    y_train = y[0:40000]
    X_test = X[40000:60000]
    y_test = y[40000:60000]
    C_param = 2 ** log_C
    #gamma = 2 ** log_gamma
    #size = int(10 ** size)
    #print "number of complete data points: " + str(X_train.shape)
    #X_train = X_train[0:size]
    #y_train = y_train[0:size]
    print "number of chosen data points: " + str(X_train.shape)

    print "number of data points test: " + str(X_test.shape)

    #clf = SVC(C=C, kernel='rbf', gamma=gamma)
    clf = LinearSVC(C=C_param)

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print "Mean Accuracy: " + str(score)

    return 1 - score


def main(job_id, params):
    print 'Parameters: '
    print params
    #return evaluation(params['Size'], params['C'], params['gamma'])
    #return evaluation(params['Size'], params['C'])
    return evaluation(params['C'])
