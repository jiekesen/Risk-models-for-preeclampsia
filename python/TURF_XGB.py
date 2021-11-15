#coding:utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import matplotlib.pyplot as plt
from time import time
import datetime
from xgboost import XGBClassifier
# from sklearn.externals.joblib import dump, load


def XGB(X_train, y_train, X_test, y_test):
    wh_xgb = XGBClassifier(objective="multi:softmax"
                    ,num_class=9
                    ,silent=0
                    ,subsample=1
                    ,gamma=0
                    ,n_estimators=10
                    ,reg_lambda=1
                    ,eval_metric='mlogloss'
                    ,learning_rate=0.225
                    )
    # grid.py
    # C = list(np.linspace(1,7,15))
    # gamma = list(np.logspace(-10,1,50))
    # n_estimators = [i for i in range(100,160,10)]
    # max_depth = [i for i in range(6,11,1)]
    n_estimators = [7,10]
    max_depth = [130,150]



    parameters = {"n_estimators": n_estimators
         ,"max_depth": max_depth}


    clf = GridSearchCV(wh_xgb, parameters, cv=5,
                       scoring="accuracy", return_train_score=False, n_jobs=40)
    #
    # scoring = {'Accuracy': make_scorer(accuracy_score)}
    # # grid parameters
    # clf = GridSearchCV(lpf_svm, parameters, cv=5, iid=True,
    #                    scoring=scoring, refit='Accuracy', return_train_score=False, n_jobs=5)

    # training model
    clf.fit(X_train, y_train)

    #
    # print(clf.cv_results_['Accuracy'][clf.best_index_])

    train_accuracy = clf.best_score_

    predict_test = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, predict_test)

    # dump(clf, "cv2_SVM.joblib")
    # predict_test.
    return train_accuracy, test_accuracy, clf.best_params_



# def scale_data(X_train, X_test):
#     min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = min_max_scaler.fit(X_train)
#     X_train_ = scaler.transform(X_train)
#     X_test_ = scaler.transform(X_test)
#     return X_train_, X_test_


def feature_overall(X_train, y_train, X_test, y_test, step_start, step_end,
                    step):
    accuracy_data = pd.DataFrame(columns=["feature number", "train_accuracy",
                                          "test_accuracy", "best_parameters"])
    for i in range(step_start, step_end, step):
        X_train_ = X_train[:, :i]
        X_test_ = X_test[:, :i]
        # X_train_, X_test_ = scale_data(X_train_, X_test_)
        feature_number = X_train_.shape[1]
        train_accuracy, test_accuracy, best_params_ = \
            XGB(X_train_, y_train, X_test_, y_test)
        print(str(feature_number) + "  train_set: " +
              str(train_accuracy) + ",test_set : " + str(test_accuracy))
        accuracy_data.loc[i] = feature_number, train_accuracy, test_accuracy, \
                               best_params_
    accuracy_data.to_csv(r'csvturf_{}-{}_xgb_accuracy.csv'.format(step_start, step_end),
                         index=None)


if __name__ == '__main__':
    train_data = pd.read_csv('/data/wanghao/pla/fea_sele/turf_x_train.csv', delimiter=',', encoding='utf-8', index_col=0, header=None)
    test_data = pd.read_csv('/data/wanghao/pla/fea_sele/turf_x_test.csv', delimiter=',', encoding='utf-8', index_col=0, header=None)

    X_train = np.array(train_data)
    X_test = np.array(test_data)
    y_train = np.array(train_data.index).ravel()
    y_test = np.array(test_data.index).ravel()


    0-500
    step_start = 10
    step_end = 1000
    step = 50
    #
    feature_overall(X_train, y_train, X_test, y_test, step_start=step_start,
                    step_end=step_end, step=step)

    # 500-1000
    step_start = 1000
    step_end = 2000
    step = 100
    #
    feature_overall(X_train, y_train, X_test, y_test, step_start=step_start,
                    step_end=step_end, step=step)

    # 1000-5000
    step_start = 2000
    step_end = 5000
    step = 200
    #
    feature_overall(X_train, y_train, X_test, y_test, step_start=step_start,
                    step_end=step_end, step=step)
    
    5000-10000
    step_start = 8000
    step_end = 8874
    step = 874
    # #
    feature_overall(X_train, y_train, X_test, y_test, step_start=step_start,
                    step_end=step_end, step=step)

    # 10000-20000
    # step_start = 10000
    # step_end = 21981
    # step = 2000
    # #
    # feature_overall(X_train, y_train, X_test, y_test, step_start=step_start,
    #                 step_end=step_end, step=step)
