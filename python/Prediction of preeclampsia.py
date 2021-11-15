#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn import metrics
# from sklearn.externals.joblib import dump, load


train_data = pd.read_csv('/data/wanghao/pla/pe/sc_1se_train.csv', delimiter=',', encoding='utf-8', index_col=0)
test_data = pd.read_csv('/data/wanghao/pla/pe/sc_1se_test.csv', delimiter=',', encoding='utf-8', index_col=0)

X_train = np.array(train_data)
X_test = np.array(test_data)
y_train = np.array(train_data.index).ravel()
y_test = np.array(test_data.index).ravel()


svm = SVC(decision_function_shape="ovo", random_state=0, kernel='rbf',C=1,probability=True)
svm.fit(X_train, y_train)

svm_pre = svm.predict(X_test)



nn = MLPClassifier(solver='adam', alpha=1e-5,
                     hidden_layer_sizes=(256,256,256)
                   , random_state=42
                   ,activation="relu"
                   ,learning_rate_init= 0.001
                   ,batch_size=256)
nn.fit(X_train, y_train)
nn_pre = nn.predict(X_test)


rfc = RandomForestClassifier(random_state=42,criterion='gini'
                             ,max_depth=14,n_estimators=280,n_jobs=10)
rfc.fit(X_train, y_train)
rfc_pre = rfc.predict(X_test)



xgb = XGBClassifier(objective="multi:softmax"
                ,num_class=9
                ,silent=0
                ,subsample=1
                ,gamma=0
                ,n_estimators=10
                ,reg_lambda=1
                ,eval_metric='mlogloss'
                ,learning_rate=0.225
                )
xgb.fit(X_train, y_train)
xgb_pre = xgb.predict(X_test)



lr = LogisticRegression(penalty='l2',
                                solver='liblinear',
                                C=0.85,
                                max_iter=300,
                                class_weight="balanced",
                                random_state=13,
                                tol=1e-4)



sclf = StackingClassifier(classifiers=[rfc,svm,xgb,nn],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr,
                         fit_base_estimators=True)
sclf.fit(X_train, y_train)


