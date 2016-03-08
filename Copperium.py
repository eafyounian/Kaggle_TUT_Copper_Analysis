#!/bin/env python

"""
A tool for predicting the class label of copper images written for 
participating in TUT Copper Analysis Competition.

Usage:
  Copperium predict <training_path> <train.csv_path> <test_path> <test.csv_path> [-r N] [-n N] [-C N]
  Copperium retrain <training_path> <train.csv_path> <test_path> <test.csv_path> <prediction_path> [-t N] [-r N] [-n N] [-C N]
  
Options:
  -h --help           Show this screen.
  -t --threshold=N    Inclusion threshold to accept a segment mean as deleted or amplified [default: 0.7]
  -r --radius=N       Radius for LBP  [default: 3]
  -n --n_points=N     Number of points for LBP  [default: 8]
  -C --C_param=N      Parameter C of LogisticRegression [default: 1]
  
Author: Ebrahim Afyounian <ebrahim.afyounian@staff.uta.fi>
"""

import numpy as np, docopt
import matplotlib.pyplot as plt
import skimage.color as color
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

def extract_features(data_path, data_csv_path, radius, n_points, threshold, is_train=True, is_retrain=False):
    X = []; y = []
    file = open(data_csv_path, 'r')
    header = next(file)
    for line in file:
        if is_retrain: 
            fname, label, prob = line.strip().split('\t')
            if float(prob) <= threshold: continue
        elif is_train:
            fname, label = line.strip().split(',')
        else:
            fname = line.strip()
        img = plt.imread(data_path + fname)
        img_gray = color.rgb2grey(img)
        lbp = local_binary_pattern(img_gray, n_points, radius, method='default')
        X.append(np.histogram(lbp, bins=np.arange(2**n_points))[0])
        if not is_train: continue
        y.append(int(label))
    file.close()
    return np.array(X), np.array(y)
    
def predict(X_train,  y_train, X_test, test_csv_path, C):
    lr = LogisticRegression(C=C)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    pred_probas = lr.predict_proba(X_test)
    file = open(test_csv_path, 'r')
    header = next(file)
    print('Id\tPrediction\tprobs')
    for i, f in enumerate(file):
        print("%s\t%s\t%s" %(f.strip(),predictions[i], np.round(np.max(pred_probas[i]),2)))
    file.close()

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    training_path = args['<training_path>']
    train_csv_path = args['<train.csv_path>']
    test_path = args['<test_path>']
    test_csv_path = args['<test.csv_path>']
    radius = int(args['--radius'])
    n_points = int(args['--n_points'])
    C = float(args['--C_param'])
    threshold = float(args['--threshold'])
    if args['predict']:
        X_train, y_train = extract_features(training_path, train_csv_path, radius, n_points, threshold, is_train=True, is_retrain=False)
        # print(X_train.shape, y_train.shape)
        X_test, y_test = extract_features(test_path, test_csv_path, radius, n_points, threshold, is_train=False, is_retrain=False)
        predict(X_train,  y_train, X_test, test_csv_path, C)
    elif args['retrain']:
        prediction_path = args['<prediction_path>']
        X_train, y_train = extract_features(training_path, train_csv_path, radius, n_points, threshold, is_train=True, is_retrain=False)
        X_train_xtra, y_train_xtra = extract_features(test_path, prediction_path, radius, n_points, threshold, is_train=True, is_retrain=True)
        X_train = np.concatenate((X_train, X_train_xtra), axis=0)
        y_train = np.concatenate((y_train, y_train_xtra), axis=0)
        X_test, y_test = extract_features(test_path, test_csv_path, radius, n_points, threshold, is_train=False, is_retrain=False)
        predict(X_train,  y_train, X_test, test_csv_path, C)
