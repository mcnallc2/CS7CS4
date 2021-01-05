import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json_lines
import random
import csv
import ftfy
import math

from google_trans_new import google_translator
from transliterate import translit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn import metrics


def get_data():
    ##
    data = pd.read_csv("reviews.csv")       # read in csv
    text = data.iloc[:, 0].tolist()         # parse text column
    voted_up = data.iloc[:, 1].tolist()     # parse voted_up column
    early_access = data.iloc[:, 2].tolist() # parse early_access column
    ##
    return(text, voted_up, early_access)


def plot_error_vs_folds(fig, X, y, model_type, max_df, K, C):
    ##
    print(f"Validating folds for {model_type}")
    fold_range = [2, 5, 10, 25, 50, 100]    # set range of folds
    fold_vals = ['2', '5', '10', '25', '50', '100']
    mean_mse = []   # init empty lists
    var_mse = []
    std_mse = []

    for folds in fold_range:    # loop through folds
        print(f"- [{folds}] of {fold_range}")
        mse, var, std = cross_val_model(X, y, model_type, folds, max_df, K, C)  # run cross val
        mean_mse.append(mse)    # add average mse to list
        var_mse.append(var)     # add mse varience to list
        std_mse.append(std)     # add mse std to list

    plt.figure(fig)
    plt.errorbar(fold_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
    plt.title(f'Mean MSE vs K-folds')
    plt.xlabel('K-folds')
    plt.ylabel('Mean MSE')
    plt.legend()
    plt.savefig(f'plots/{model_type}_kfold_crossval_2.png')
    plt.show()


def plot_error_vs_max_df(fig, X, y, model_type, folds, K, C):
    ##
    print(f"Validating max_df for {model_type}")
    max_df_range = [0.01, 0.1, 0.3, 0.5, 0.8, 1]    # set range of max_df
    max_df_vals = ['0.01', '0.1', '0.3', '0.5', '0.8', '1']
    mean_mse = []   # init empty lists
    var_mse = []
    std_mse = []

    for max_df in max_df_range:    # loop through folds
        print(f"- [{max_df}] of {max_df_range}")
        mse, var, std = cross_val_model(X, y, model_type, folds, max_df, K, C)  # run cross val
        mean_mse.append(mse)    # add average mse to list
        var_mse.append(var)     # add mse varience to list
        std_mse.append(std)     # add mse std to list

    plt.figure(fig)
    plt.errorbar(max_df_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
    plt.title(f'Mean MSE vs max_df')
    plt.xlabel('max_df')
    plt.ylabel('Mean MSE')
    plt.legend()
    plt.savefig(f'plots/{model_type}_max_df_crossval_2.png')
    plt.show()


def plot_error_vs_C(fig, X, y, model_type, max_df, folds):
    ##
    print(f"Validating C for {model_type}")
    C_range = [0.01, 0.1, 1, 10, 100]    # set range of C
    C_vals = ['0.01', '0.1', '1', '10', '100']
    K = 'N/A'
    mean_mse = []   # init empty lists
    var_mse = []
    std_mse = []

    for C in C_range:
        print(f"- [{C}] of {C_range}")
        mse, var, std = cross_val_model(X, y, model_type, folds, max_df, K, C)  # run cross val
        mean_mse.append(mse)    # add average mse to list
        var_mse.append(var)     # add mse varience to list
        std_mse.append(std)     # add mse std to list

    plt.figure(fig)
    plt.errorbar(C_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
    plt.title(f'Mean MSE vs C penalty')
    plt.xlabel('C penalty')
    plt.ylabel('Mean MSE')
    plt.legend()
    plt.savefig(f'plots/{model_type}_C_penalty_crossval_2.png')
    plt.show()


def plot_error_vs_K(fig, X, y, model_type, max_df, folds):
    ##
    print(f"Validating K for {model_type}")
    K_range = [2, 5, 10, 20, 50, 75, 100]    # set range of K
    K_vals = ['2', '5', '10', '20', '50', '75', '100']
    C = 'N/A'
    mean_mse = []   # init empty lists
    var_mse = []
    std_mse = []

    for K in K_range:
        print(f"- [{K}] of {K_range}")
        mse, var, std = cross_val_model(X, y, model_type, folds, max_df, K, C)  # run cross val
        mean_mse.append(mse)    # add average mse to list
        var_mse.append(var)     # add mse varience to list
        std_mse.append(std)     # add mse std to list

    plt.figure(fig)
    plt.errorbar(K_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
    plt.title(f'Mean MSE vs KNN')
    plt.xlabel('KNN')
    plt.ylabel('Mean MSE')
    plt.legend()
    plt.savefig(f'plots/{model_type}_KNN_crossval_2.png')
    plt.show()

def cross_val_model(X, y, model_type, folds, max_df, K, C):
    ##
    mse=[]
    kf = KFold(n_splits=folds)  # fold dataset
    f=0

    for train, test in kf.split(X):
        f+=1
        print(f"-- {f} of {folds}")
        ##
        X_train=[]; y_train=[]  # converting data to python list format
        for i in train:
            X_train.append(X[i])
            y_train.append(y[i])
        
        X_test=[]; y_test=[]
        for i in test:
            X_test.append(X[i])
            y_test.append(y[i])
    
        vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df)   # create vectorizor
        tfidf_train = vectorizer.fit_transform(X_train).toarray()   # vectorize training data
        tfidf_test = vectorizer.transform(X_test).toarray()         # vectorize test data

        if model_type == 'LR':
            model = LogisticRegression(C=C, penalty='l1', solver='saga', max_iter=90000)    # create LR model
        elif model_type == 'SVM':
            model = LinearSVC(C=C, max_iter=90000)  # create SVM model
        else:
            model = KNeighborsClassifier(n_neighbors=K, weights='uniform')  # create kNN model

        model.fit(tfidf_train, y_train) # train the specified model
        predicted = model.predict(tfidf_test) # get model predictions

        mse.append(metrics.mean_squared_error(y_test, predicted)) # append mse of predictions to list

    return np.mean(mse), np.var(mse), np.std(mse)   # return mean mse, mse varience, mse std


def tfidf_modelling(X_train, X_test, y_train, y_test, model_type, max_df, K, C):
    ##
    vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df)   # create vectorizor
    tfidf_train = vectorizer.fit_transform(X_train).toarray()   # vectorize training data
    tfidf_test = vectorizer.transform(X_test).toarray()         # vectorize test data

    if model_type == 'LR':
        model = LogisticRegression(C=C, penalty='l1', solver='saga', max_iter=90000)    # create LR model
    elif model_type == 'SVM':
        model = LinearSVC(C=C, max_iter=90000)  # create SVM model
    else:
        model = KNeighborsClassifier(n_neighbors=K, weights='uniform')  # create kNN model

    model.fit(tfidf_train, y_train) # train the specified model
    predicted = model.predict(tfidf_test) # get model predictions
    
    if model_type == 'SVM':
        fpr, tpr, _ = metrics.roc_curve(y_test, model._predict_proba_lr(tfidf_test)[:,1]) # get false postive and true postive values for roc plot
    else:
        fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(tfidf_test)[:,1]) # get false postive and true postive values for roc plot

    acc = metrics.accuracy_score(y_test, predicted)     # get accuracy score
    mse = metrics.mean_squared_error(y_test, predicted) # get mse value
    cm = metrics.confusion_matrix(y_test, predicted)    # get confusion matrix
    auc = metrics.roc_auc_score(y_test, predicted)      # roc auc

    print(f'\nDummy with TFIDF')
    print(f'Accuracy = ' + str(acc * 100) + '%')
    print(f'MSE = {mse}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'ROC AUC = {auc}')
    print(f'Confusion Matrix:\n{cm}')

    return (fpr, tpr)


def dummy_modelling(X_train, X_test, y_train, y_test):
    ##
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)   # create vectorizor
    tfidf_train = vectorizer.fit_transform(X_train).toarray()   # vectorize training data
    tfidf_test = vectorizer.transform(X_test).toarray()         # vectorize test data

    ## select specified model
    dummy = DummyClassifier(strategy='most_frequent') # create dummy model

    dummy.fit(tfidf_train, y_train) # train the dummy model
    predicted = dummy.predict(tfidf_test) # get model predictions

    fpr, tpr, _ = metrics.roc_curve(y_test, dummy.predict_proba(tfidf_test)[:,1]) # get false postive and true postive values for roc plot
    
    acc = metrics.accuracy_score(y_test, predicted)     # get accuracy score
    mse = metrics.mean_squared_error(y_test, predicted) # get mse value
    cm = metrics.confusion_matrix(y_test, predicted)    # get confusion matrix
    auc = metrics.roc_auc_score(y_test, predicted)      # roc auc

    print(f'\nDummy with TFIDF')
    print(f'Accuracy = ' + str(acc * 100) + '%')
    print(f'MSE = {mse}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'ROC AUC = {auc}')
    print(f'Confusion Matrix:\n{cm}')

    return (fpr, tpr)



X, y, z = get_data()    # get pre-processed data from csv file


#### part (i) using text (X) to predict if game is voted up or down (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    # train/test split (X, y)

## Logistic Regression
plot_error_vs_folds(1, X_train, y_train, 'LR', 0.2, 0, 1)   # crossval folds
plot_error_vs_max_df(2, X_train, y_train, 'LR', 10, 0, 1)   # crossval max_df
plot_error_vs_C(3, X_train, y_train, 'LR', 0.8, 10)         # crossval C
(fpr1, tpr1) = tfidf_modelling(X_train, X_test, y_train, y_test, 'LR', 0.8, 0, 10)  # train and evaluate Logistic Regression model

## SVM
plot_error_vs_folds(4, X_train, y_train, 'SVM', 0.2, 0, 1)  # crossval folds
plot_error_vs_max_df(5, X_train, y_train, 'SVM', 5, 0, 1)   # crossval max_df
plot_error_vs_C(6, X_train, y_train, 'SVM', 0.8, 5)         # crossval C
(fpr2, tpr2) = tfidf_modelling(X_train, X_test, y_train, y_test, 'SVM', 0.8, 0, 1)  # train and evaluate SVM model

## kNN
plot_error_vs_folds(7, X_train, y_train, 'KNN', 0.2, 3, 0)   # crossval folds
plot_error_vs_max_df(8, X_train, y_train, 'KNN', 10, 3, 0)   # crossval max_df
plot_error_vs_K(9, X_train, y_train, 'KNN', 0.1, 10)         # crossval K
(fpr3, tpr3) = tfidf_modelling(X_train, X_test, y_train, y_test, 'KNN', 0.1, 10, 0) # train and evaluate kNN model

## dummy classifer
(fpr4, tpr4) = dummy_modelling(X_train, X_test, y_train, y_test)    # train and evaluate dummy classifier

## plot the ROC curves
plt.figure(10)
plt.plot(fpr1, tpr1, label='Logistic Regression')
plt.plot(fpr2, tpr2, label='SVM')
plt.plot(fpr3, tpr3, label='kNN')
plt.plot(fpr4, tpr4, label='Baseline (most frequent)')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='45 degree')
plt.title('ROC Curves')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig(f'plots/ROC_curve_1.png')
plt.show()



#### part (ii) using text (X) to predict if its an early access version of the game (z)

X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)    # train/test split (X, z)

## Logistic Regression
plot_error_vs_folds(11, X_train, y_train, 'LR', 0.2, 0, 1)   # crossval folds
plot_error_vs_max_df(12, X_train, y_train, 'LR', 5, 0, 1)    # crossval max_df
plot_error_vs_C(13, X_train, y_train, 'LR', 0.01, 5)         # crossval C
(fpr5, tpr5) = tfidf_modelling(X_train, X_test, y_train, y_test, 'LR', 0.01, 0, 1)

## SVM
plot_error_vs_folds(14, X_train, y_train, 'SVM', 0.2, 0, 1)  # crossval folds
plot_error_vs_max_df(15, X_train, y_train, 'SVM', 5, 0, 1)   # crossval max_df
plot_error_vs_C(16, X_train, y_train, 'SVM', 1.0, 5)         # crossval C
(fpr6, tpr6) = tfidf_modelling(X_train, X_test, y_train, y_test, 'SVM', 1.0, 0, 0.01)

## KNN
plot_error_vs_folds(17, X_train, y_train, 'KNN', 0.2, 3, 0)  # crossval folds
plot_error_vs_max_df(18, X_train, y_train, 'KNN', 10, 3, 0)  # crossval max_df
plot_error_vs_K(19, X_train, y_train, 'KNN', 0.5, 10)        # crossval K
(fpr7, tpr7) = tfidf_modelling(X_train, X_test, y_train, y_test, 'KNN', 0.5, 10, 0)

## dummy classifier
(fpr8, tpr8) = dummy_modelling(X_train, X_test, y_train, y_test)    # train and evaluate dummy classifier

## plot the ROC curves
plt.figure(20)
plt.plot(fpr5, tpr5, label='Logistic Regression')
plt.plot(fpr6, tpr6, label='SVM')
plt.plot(fpr7, tpr7, label='kNN')
plt.plot(fpr8, tpr8, label='Baseline (most frequent)')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='45 degree')
plt.title('ROC Curves')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig(f'plots/ROC_curve_2.png')
plt.show()