#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

## creating dummy training set
Xtrain = np.array([-1.0, 0.0, 1.0]); Xtrain = Xtrain.reshape(-1,1)
ytrain = np.array([ 0.0, 1.0, 0.0]); ytrain = ytrain.reshape(-1,1)

## read and format data in csv file
df = pd.read_csv("week5.csv",comment='#',header=None)
X = np.array(df.iloc[:,0]); X = X.reshape(-1,1)
y = np.array(df.iloc[:,1]); y = y.reshape(-1,1)

## function to calculate the gaussian kernal weights
def gaussian_kernel(distances):
    weights = np.exp(-gamma*(distances**2))
    return weights/np.sum(weights)

## function to run cross validation 
def cross_val_model(mod_type, folds, gamma, C):
    ##
    ## init mean-squared-error array
    mse=[]
    ## 
    ## make k-fold obj
    kf = KFold(n_splits=folds)
    ##
    ## loop through each k-fold split
    for train, test in kf.split(X):
        ##
        ## if generating KNN model
        if mod_type == 'knn':
            model = KNeighborsRegressor(n_neighbors=len(X[train]), weights=gaussian_kernel).fit(X[train], y[train])
        ##
        ## if generating Ridge Reg model
        else:
            model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(X, y)
        ##
        ## get pridictions using test part of split
        ypred = model.predict(X[test])
        ##
        ## get error for predictions and append to error list
        mse.append(mean_squared_error(y[test], ypred))
    ##
    ## return mean, varience and standard dev error values
    return [np.mean(mse), np.var(mse), np.std(mse)]


## (i)(a)
## generate range of X feature values for predictions
Xpred = np.linspace(-3.0,3.0,300); Xpred = Xpred.reshape(-1,1)
##
## plot the dummy training data
plt.figure(1)
plt.scatter(Xtrain, ytrain, label='training data')
##
## Gamma value range
gamma_range = [0, 1, 5, 10, 25]
##
## generate model for gamma value range
for gamma in gamma_range:
    ##
    ## generate kernalised knn regressor (3-NN for dummy set)
    model = KNeighborsRegressor(n_neighbors=3, weights=gaussian_kernel).fit(Xtrain, ytrain)
    ##
    ## determine target predictions and plot
    ypred = model.predict(Xpred)
    plt.plot(Xpred, ypred, label=f'Gamma={gamma}')
##
plt.title('Kernalised KNN Regression (Dummy data)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()


## (i)(c)
## great subplot of kernalised ridge reg models
plt.figure(2)
plt.suptitle('Kernalised Ridge Regression (Dummy Data))')
##
## C penalty values and Gamma values
C_range=[0.1, 1, 10, 1000]
gamma_range=[0, 1, 5, 10, 25]
##
## loop through C values
for i, C in enumerate(C_range):
    ##
    ## plot the dummy training data
    plt.subplot(2, 2, i+1)
    plt.scatter(Xtrain, ytrain, label='training data')
    plt.title(f"C = {C}")
    ##
    ## loop through Gamma values
    for gamma in gamma_range:
        ##
        ## generate Ridge regressors
        model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(Xtrain, ytrain)
        ##
        ## determine target predictions and add to subplot
        ypred = model.predict(Xpred)
        plt.plot(Xpred, ypred, label=f'Gamma={gamma}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        print(f"Kernalised Ridge Regressor (C={C}, Gamma={gamma})")
        print(f"dual_coef_:\n{model.dual_coef_}")
    ##
##


## (ii)
## plot the training data
# plt.figure(3)
# plt.scatter(X, y, label='training data')
# plt.title("Training Data")
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()

## (ii)(a)
## plot training data
plt.figure(4)
plt.scatter(X, y, label='training data')
##
## set gamma value range
gamma_range=[0, 1, 5, 10, 25, 100]
##
## loop through gamma values
for gamma in gamma_range:
    ##
    ## generate kernalised knn regressors
    model = KNeighborsRegressor(n_neighbors=len(X), weights=gaussian_kernel).fit(X, y)
    ##
    ## determine target predictions and plot
    ypred = model.predict(Xpred)
    plt.plot(Xpred, ypred, label=f'Gamma={gamma}')
##
plt.title('Kernalised KNN Regression (Training Data)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()


## (ii)(b)
## plot training data
plt.figure(5)
plt.scatter(X, y, label='training data')
##
## use C=1000
C=1000
##
## set gamma range
gamma_range=[0, 1, 5, 10, 25, 100]
##
## loop through gamma values
for gamma in gamma_range:
    ##
    ## generate Kernalised Ridge Regressors
    model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(X, y)
    ##
    ## determine target predictions and plot
    ypred = model.predict(Xpred)
    plt.plot(Xpred, ypred, label=f'Gamma={gamma}')
##
plt.title(f'Kernalised Ridge Regression (Training Data) C={C}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()


## (ii)(C)
## cross validation for kernalised KNN regressor
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
    ##
    ## run cross validation for Kernalised Ridge Regressors (Gamma=1)
    cross_val_results = cross_val_model('knn', folds, 1, 'NA')
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(6)
kf = ['2', '10', '25', '50', '100']
plt.errorbar(kf, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Kernalised KNN Regression mean prediction error vs k-fold values (Gamma=1)')
plt.xlabel('k-fold value')
plt.ylabel('Mean prediction error')
plt.legend()
##
##
## init list of C values
gamma_range=[0, 1, 5, 10, 25, 100, 200, 500]
##
## init mean, var and std lists
mean_mse = []
var_mse = []
std_mse = []
##
## for each C value
for gamma in gamma_range:
    ##
    ## run cross validation for Kernalised Ridge Regressors (k-folds=2)
    cross_val_results = cross_val_model('knn', 2, gamma, 'NA')
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(7)
gamma_vals=['0', '1', '5', '10', '25', '100', '200', '500']
plt.errorbar(gamma_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Kernalised KNN Regression mean prediction error vs Gamma values (k-folds = 2)')
plt.xlabel('Gamma value')
plt.ylabel('Mean prediction error')
plt.legend()
##
##
## plot tuned kernalised KNN regressor
##
## plot training data
plt.figure(8)
plt.scatter(X, y, label='training data')
##
gamma=100
## generate Kernalised Ridge Regressor
model = KNeighborsRegressor(n_neighbors=len(X), weights=gaussian_kernel).fit(X, y)
##
## determine target predictions and plot
ypred = model.predict(Xpred)
plt.plot(Xpred, ypred, color='red', label=f'Gamma={gamma}')
##
plt.title(f'Kernalised Ridge Regression (Training Data) Gamma={gamma}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
##
##
##
## cross validation for kernalised KNN regressor
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
    ##
    ## run cross validation for Kernalised Ridge Regressior (Gamma=1, C=1)
    cross_val_results = cross_val_model('ridge', folds, 1, 1)
    ##
    ## append append mean, var and std error values to lists
    mean_mse.append(cross_val_results[0])
    var_mse.append(cross_val_results[1])
    std_mse.append(cross_val_results[2])
##
## add results to errorbar plot
plt.figure(9)
kf = ['2', '10', '25', '50', '100']
plt.errorbar(kf, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
plt.title('Kernalised Ridge Regression mean prediction error vs k-fold values (Gamma=1, C=1)')
plt.xlabel('k-fold value')
plt.ylabel('Mean prediction error')
plt.legend()
##
##
plt.figure(10)
## init list of C values
C_range = [1, 10, 100, 1000]
gamma_range = [1, 5, 10, 25, 100, 500]
gamma_vals = ['1', '5', '10', '25', '100', '500']#, '1500', '2000','5000', '10000']
##
## init mean, var and std lists
mean_mse = []
var_mse = []
std_mse = []
##
## loop through C values
for C in C_range:
    mean_mse = []
    var_mse = []
    std_mse = []
    ##
    ## for each C value
    for gamma in gamma_range:
        ##
        ## run cross validation for Kernalised Ridge Regressors
        cross_val_results = cross_val_model('ridge', 10, gamma, C)
        ##
        ## append append mean, var and std error values to lists
        mean_mse.append(cross_val_results[0])
        var_mse.append(cross_val_results[1])
        std_mse.append(cross_val_results[2])
    ##
    plt.errorbar(gamma_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label=f'C = {C}')
##
plt.title('Kernalised Ridge Regression mean prediction error vs Gamma values')
plt.xlabel('Gamma value')
plt.ylabel('Mean prediction error')
plt.legend()
##
##
## plot tuned kernalised Ridge regressor
##
## plot training data
plt.figure(11)
plt.scatter(X, y, label='training data')
##
## set tuned hyperparams
C=100; gamma=5
##
## generate Kernalised Ridge Regressor
model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(X, y)
##
## determine target predictions and plot
ypred = model.predict(Xpred)
plt.plot(Xpred, ypred, color='red', label=f'Predictions')
##
plt.title(f'Kernalised Ridge Regression (Training Data) (Gamma={gamma}, C={C})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
##
## plot training data
plt.figure(12)
plt.scatter(X, y, label='training data')
##
## set tuned hyperparams
C=1000; gamma=500
##
## generate Kernalised Ridge Regressor
model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(X, y)
##
## determine target predictions and plot
ypred = model.predict(Xpred)
plt.plot(Xpred, ypred, color='red', label=f'Predictions')
##
plt.title(f'Kernalised Ridge Regression (Training Data) (Gamma={gamma}, C={C})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

## display plots
plt.show()