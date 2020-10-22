#!/usr/bin/python

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv("week3.csv",comment='#',header=None)
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]

## (i) (a)
##
## 3D scatter plot of training data
plt.figure(1)
ax = plt.gca(projection='3d')
ax.scatter(X[:,0], X[:,1], y, label='Training Data')
plt.title('3D scatter plot of training data (2 features)')
ax.set_xlabel('X_1 features')
ax.set_ylabel('X_2 features')
ax.set_zlabel('Target Values')
plt.legend()

## function to tune hyperparameters
def tune_hyperparams(model_type, fig_shift):
    ##
    ## init Q power and C penalty values
    q_range = [1, 2, 3, 4, 5]
    c_range = [0.0001, 1, 10, 1000]
    ##
    ## (i) (b)
    ##
    ## loop through Q and C values
    for Q in q_range:
        for C in c_range:
            ##
            ## generate more features of higher powers
            Xpoly = PolynomialFeatures(Q).fit_transform(X)
            # Xpoly = PolynomialFeatures(Q).fit(X)
            # print(f'{model_type} model (Order={Q}, C={C})')
            # print(Xpoly.get_feature_names())
            ##
            ## if generating Lasso model
            if model_type == 'Lasso':
                model = linear_model.Lasso(alpha=1/(2*C)).fit(Xpoly, y)
            ##
            ## generate Ridge model
            else:
                model = linear_model.Ridge(alpha=1/(2*C)).fit(Xpoly, y)
            ##
            ## determine target predictions
            y_pred = model.predict(Xpoly)
            ##
            print(f'{model_type} model (Q={Q}, C={C})')
            print(f'Paramters = {model.coef_}\n')
        ##
    ##
    ##
    ## (i) (c)
    ##
    ## init emtpy array
    Xtest = []
    ##
    ## init array of values between -5,5
    grid = np.linspace(-5,5)
    ##
    ## push values into 2D array
    for i in grid:
        for j in grid:
            Xtest.append([i,j])
        ##
    ##
    ## split into 2D arrays for each feature
    Xtest = np.array(Xtest)
    x1 = np.reshape(Xtest[:,0], (50, 50))
    x2 = np.reshape(Xtest[:,1], (50, 50))
    ##
    ## subplot position dictionary
    subplot = {1:221,2:222,3:223,4:224}
    ##
    ## loop through Q values
    for i, Q in enumerate(q_range):
        ## set fig number
        fig = plt.figure(i+2+fig_shift)
        flag=0
        ##
        ## loop through C values
        for j, C in enumerate(c_range):
            ##
            ## generate more features of higher powers for training and test data
            X_poly = PolynomialFeatures(Q).fit_transform(X)
            X_test = PolynomialFeatures(Q).fit_transform(Xtest)
            ##
            ## if generating Lasso model
            if model_type == 'Lasso':
                model = linear_model.Lasso(alpha=1/(2*C)).fit(X_poly, y)
            ##
            ## else generate ridge model
            else:
                model = linear_model.Ridge(alpha=1/(2*C)).fit(X_poly, y)
            ##
            ## generate predictions using test feature data
            y_pred = np.reshape(model.predict(X_test), (50, 50))
            ##
            ## add to subplot
            ax = fig.add_subplot(subplot[j+1], projection='3d')
            ax.scatter(X[:,0], X[:,1], y, label='Training Data' if flag == 0 else '')
            ax.plot_wireframe(x1, x2, y_pred, color='grey', rstride=3, cstride=3, linewidth=1, label='Predictions' if flag == 0 else '')
            ax.set_title(f'Penalty C={C}')
            ax.set_xlabel('X_1 features')
            ax.set_ylabel('X_2 features')
            ax.set_zlabel('Target Values')
            flag=1
        ##
        fig.legend()
        fig.suptitle(f'{model_type} regression model with 2 features up to power Q={Q}')
    ##

## (i) (b) (c)
##
## tune hyperparamters for Lasso models
tune_hyperparams('Lasso', 0)
##
## (i) (e)
##
## tune hyperparamters for Ridge models
tune_hyperparams('Ridge', 5)


## function to run cross validation 
def cross_val_model(model_type, folds, Q, C):
    ##
    ## generate more features of higher powers for training
    Xpoly = PolynomialFeatures(Q).fit_transform(X)
    ##
    ## int mean-squared-error array
    mse=[]
    ## 
    ## make k-fold obj
    kf = KFold(n_splits=folds)
    ##
    ## loop through each k-fold split
    for train, test in kf.split(Xpoly):
        ##
        ## if generateing Lasso model
        if model_type == 'Lasso':
            model = linear_model.Lasso(alpha=1/(2*C)).fit(Xpoly[train], y[train])
        ##
        ## else generate Ridge model
        else:
            model = linear_model.Ridge(alpha=1/(2*C)).fit(Xpoly[train], y[train])
        ##
        ## get pridictions using test part of split
        ypred = model.predict(Xpoly[test])
        ##
        ## get error for predictions and append to error list
        mse.append(mean_squared_error(y[test], ypred))
        #print(f'intercept {model.intercept_}, slope {model.coef_}, mse {mean_squared_error(y[test], ypred)}')  
    ##
    ## return mean, varience and standard dev error values
    return [np.mean(mse), np.var(mse), np.std(mse)]
    ##
    # print(f'k-fold cross-validation (folds {folds})')
    # print(f'Mean MSE - {np.mean(mse)}')
    # print(f'Var MSE - {np.var(mse)}')
    # print(f'Std MSE - {np.std(mse)}\n')


## (ii) (a)
##
## get error values for 5-fold
cross_val_results = cross_val_model('Lasso', 5, 2, 1)
##
## print mean, var and std error values
print(f'5-fold cross-validation')
print(f'Mean MSE - {cross_val_results[0]}')
print(f'Var MSE - {cross_val_results[1]}')
print(f'Std MSE - {cross_val_results[2]}\n')
##
## init list of diffent k-fold values
k_folds = [2, 10, 25, 50, 100]
##
## init mean, var and std lists
mean_mse = []
var_mse = []
std_mse = []
##
## for each k-fold value
for folds in k_folds:
    ## run cross validation for Lasso model for Q=2, C=1
    cross_val_results = cross_val_model('Lasso', folds, 2, 1)
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(12)
kf = ['2', '10', '25', '50', '100']
# print(mean_mse)
# print(var_mse)
plt.errorbar(kf, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Lasso mean prediction error vs k-fold values (Q = 2, C = 1)')
plt.xlabel('k-fold value')
plt.ylabel('Mean prediction error')
plt.legend()


## (ii) (b)
##
## init list of C values
c_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
##
## init mean, var and std lists
mean_mse = []
var_mse = []
std_mse = []
##
## for each C value
for C in c_vals:
    ## run cross validation for Lasso model for k-folds=10, Q=2
    cross_val_results = cross_val_model('Lasso', 10, 2, C)
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(13)
c_vals = ['0.01', '0.1', '1', '10', '100', '1000', '10000']
# print(mean_mse)
# print(var_mse)
plt.errorbar(c_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Lasso mean prediction error vs C penalty values (Q = 2, k-folds = 10)')
plt.xlabel('C penalty value')
plt.ylabel('Mean prediction error')
plt.legend()


## (ii) (d)
##
## init list of diffent k-fold values
k_folds = [2, 10, 25, 50, 100]
##
## init mean, var and std lists
mean_mse = []
var_mse = []
std_mse = []
##
## for each k-fold value
for folds in k_folds:
    ## run cross validation for Ridge model for Q=2, C=1
    cross_val_results = cross_val_model('Ridge', folds, 2, 1)
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(14)
kf = ['2', '10', '25', '50', '100']
# print(mean_mse)
# print(var_mse)
plt.errorbar(kf, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Ridge mean prediction error vs k-fold values (Q = 2, C = 1)')
plt.xlabel('k-fold value')
plt.ylabel('Mean prediction error')
plt.legend()


## init list of C values
c_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
##
## init mean, var and std lists
mean_mse = []
var_mse = []
std_mse = []
##
## for each C value
for C in c_vals:
    ## run cross validation for Ridge model for k-fold=10, Q=2
    cross_val_results = cross_val_model('Ridge', 10, 2, C)
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(15)
c_vals = ['0.01', '0.1', '1', '10', '100', '1000', '10000']
# print(mean_mse)
# print(var_mse)
plt.errorbar(c_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Ridge mean prediction error vs C penalty values (Q = 2, k-folds = 10)')
plt.xlabel('C penalty value')
plt.ylabel('Mean prediction error')
plt.legend()


## show plots
plt.show()