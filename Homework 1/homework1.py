from collections import defaultdict
from sklearn import linear_model
import numpy
import math
import gzip
import json
import dateutil.parser
import random

def getMaxLen(dataset):

    maxLen = 0
    for review in dataset:
        sample_len = len(review['review_text'])
        maxLen = max(maxLen, sample_len)

    return maxLen

def featureQ1(datum, maxLen):

    return [1, len(datum['review_text'])/maxLen]

def Q1(dataset):

    maxLen = getMaxLen(dataset)
    x = []
    y = []
    for datum in dataset:
        x.append(featureQ1(datum, maxLen))
        y.append(datum['rating'])

    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(x,y)

    yi = numpy.sum(numpy.asarray(x) * reg.coef_, axis=1)

    MSE = sum((yi - y)**2)/len(y)
    theta = reg.coef_
    return theta, MSE

def featureQ2(datum, maxLen):

    date = dateutil.parser.parse(datum['date_added'])
    weekday_features = [int(date.isoweekday() == day) for day in range(2, 8)]
    month_features = [int(date.month == month) for month in range(2, 13)]

    return [1, len(datum['review_text']) / maxLen] + weekday_features + month_features

def Q2(dataset):

        maxLen = getMaxLen(dataset)
        X2 = []
        Y2 = []
        for datum in dataset:
            X2.append(featureQ2(datum, maxLen))
            Y2.append(datum['rating'])

        reg = linear_model.LinearRegression(fit_intercept=False)
        reg.fit(X2,Y2)

        Yi = numpy.sum(numpy.asarray(X2) * reg.coef_, axis=1)

        MSE2 = sum((Yi - Y2)**2)/len(Y2)

        return X2, Y2, MSE2

def featureQ3(datum, maxLen):

    date = dateutil.parser.parse(datum['date_added'])
    return [1, len(datum['review_text']) / maxLen, int(date.weekday()), int(date.month)]

def Q3(dataset):

        maxLen = getMaxLen(dataset)
        X3 = []
        Y3 = []
        for datum in dataset:
            X3.append(featureQ3(datum, maxLen))
            Y3.append(datum['rating'])
        reg = linear_model.LinearRegression(fit_intercept=False)
        reg.fit(X3,Y3)

        Yi = numpy.sum(numpy.asarray(X3) * reg.coef_, axis=1)

        MSE3 = sum((Yi - Y3)**2)/len(Y3)

        return X3, Y3, MSE3

def Q4(dataset):
    n = len(dataset) // 2
    dataset_train, dataset_test = dataset[:n], dataset[n:]

    maxLen = getMaxLen(dataset_train)

    X2_train = [featureQ2(datum, maxLen) for datum in dataset_train]
    Y2_train = [datum['rating'] for datum in dataset_train]
    X2_test = [featureQ2(datum, maxLen) for datum in dataset_test]
    Y2_test = [datum['rating'] for datum in dataset_test]

    reg2 = linear_model.LinearRegression(fit_intercept=False)
    reg2.fit(X2_train, Y2_train)
    Y2i = numpy.sum(numpy.asarray(X2_test) * reg2.coef_, axis=1)
    test_mse2 = sum((Y2i - Y2_test)**2) / len(Y2_test)

    X3_train = [featureQ3(datum, maxLen) for datum in dataset_train]
    Y3_train = [datum['rating'] for datum in dataset_train]
    X3_test = [featureQ3(datum, maxLen) for datum in dataset_test]
    Y3_test = [datum['rating'] for datum in dataset_test]

    reg3 = linear_model.LinearRegression(fit_intercept=False)
    reg3.fit(X3_train, Y3_train)
    Y3i = numpy.sum(numpy.asarray(X3_test) * reg3.coef_, axis=1)
    test_mse3 = sum((Y3i - Y3_test)**2) / len(Y3_test)

    return test_mse2, test_mse3

def featureQ5(datum):

    x = [1, len(datum['review/text'])]
    y = int(datum['review/overall']>=4)
    return x, y

def Q5(dataset, feat_func):

    X5 = []
    Y5 = []
    for datum in dataset:
        x, y = feat_func(datum)
        X5.append(x)
        Y5.append(y)


    model = linear_model.LogisticRegression(class_weight='balanced',fit_intercept=False, max_iter=5000,)
    model.fit(X5, Y5)
    pred = model.predict(X5)

    TP = sum(numpy.logical_and(Y5, pred))
    TN = sum(numpy.logical_and(numpy.logical_not(Y5), numpy.logical_not(pred)))
    FP = sum(numpy.logical_and(numpy.logical_not(Y5), pred))
    FN = sum(numpy.logical_and(Y5, numpy.logical_not(pred)))

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)

    BER = 1 - 0.5*(TPR + TNR)

    return TP, TN, FP, FN, BER

def Q6(dataset):


    X5 = []
    Y5 = []
    for datum in dataset:
        x, y = [1, len(datum['review/text'])], int(datum['review/overall']>=4)
        X5.append(x)
        Y5.append(y)

    model = linear_model.LogisticRegression(class_weight='balanced',fit_intercept=False, max_iter=5000)
    model.fit(X5, Y5)
    pred = model.decision_function(X5)

    score_labels = list(zip(pred, Y5))
    score_labels.sort(key=lambda pair: pair[0], reverse=True)

    return tuple(
        sum(label for _, label in score_labels[:k]) / k
        for k in [1, 100, 1000, 10000]
    )

def featureQ7(datum):
    x = [1, len(datum['review/text']), datum['review/appearance'], datum['review/palate'], datum['review/taste'], datum['review/aroma'], datum['beer/ABV']]
    y = int(datum['review/overall']>=4)
    return x, y
