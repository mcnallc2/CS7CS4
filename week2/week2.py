#!/usr/bin/python

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df = pd.read_csv("week2.csv",comment='#',header=None)
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]


def plot_train_target_values(y):
    l1=l2=0
    ## iterate throug the target values
    for i, val in enumerate(y):
        ## if target value is a +1
        if val == 1:
            ## add point to the plot with a '+' marker
            plt.scatter(X[i][0], X[i][1], s=60, color='blue', marker='+', label='Training Data (+1)' if l1 == 0 else '')
            l1=1
        ##
        ## else if target value is a -1
        elif val == -1:
            ## add point to the plot with a 'o' marker
            plt.scatter(X[i][0], X[i][1], s=30, color='blue', marker='o', label='Training Data (-1)' if l2 == 0 else '')
            l2=1
        ##
    ##
    l1=l2=0

def plot_pred_target_values(pred):
    l1=l2=0
    ## iterate through the predicted target values
    for i, val in enumerate(pred):
        ## if prediction is +1
        if val == 1:
            ## add point to the plot with a '+' marker
            plt.scatter(X[i][0], X[i][1], s=30, marker='+', edgecolor='black', linewidth=0.5, facecolor='lime', label='Prediction (+1)' if l1 == 0 else '')
            l1=1
        ##
        ## if prediction is -1
        elif val == -1:
            ## add point to the plot with a 'o' marker
            plt.scatter(X[i][0], X[i][1], s=15, marker='o', edgecolor='black', linewidth=0.5, facecolor='lime', label='Prediction (-1)' if l2 == 0 else '')
            l2=1
        ##
    ##
    l1=l2=0


## (a) (i)
##
## create new figure
plt.figure(1)
## plot training data
plot_train_target_values(y)

## (a) (ii)
##
## create logistic regression model using sklearn and fit the (2 feature) data
model = LogisticRegression(penalty='none',solver='lbfgs').fit(X,y)
## print model info
print(f'Model - Logistic Regression (2 features)')
print(f'Penalty - none')
print(f'Intercept - {model.intercept_}')
print(f'Slope - {model.coef_}\n')

## (a) (iii)
##
plt.figure(2)
## plot taining data
plot_train_target_values(y)
##
## predict new target values using the model
pred = model.predict(X)
## plot model predictions
plot_pred_target_values(pred)

## the model (2 features) can be represented as
## h(x) = w0 + w1x1 + w2x2
## h(x) = 0 - @ desision boundary
## 0 = w0 + w1x1 + w2x2
## equation for x2: x2 = -(w0 + w1x1)/w2
##
## store the model parameters
t0 = model.intercept_[0]
t1 = model.coef_[0][0]
t2 = model.coef_[0][1]
##
## create a range of X_1 values
x1_bound = np.linspace(-1, 1, 100)
## determine the corresponding X_2 values
x2_bound = -(t0 + (x1_bound * t1)) / t2
## plot the desision boundary
plt.plot(x1_bound, x2_bound, '--', c="red", label='Desision Boundary')


## (b) (i)(ii)
for i, penalty in enumerate([0.001, 1, 1000]):
    ## create figure
    plt.figure(i+3)
    ## plot training data
    plot_train_target_values(y)
    ##
    ## create linear SVM model using sklearn and fit the (2 feature) data
    model = LinearSVC(C=penalty, max_iter=90000).fit(X,y)
    print(f'Model - Linear SVM (2 features)')
    print(f'Penalty - {penalty}')
    print(f'Intercept - {model.intercept_}')
    print(f'Slope - {model.coef_}\n')
    ##
    ## predict SVM model target values
    pred = model.predict(X)
    ## plot model predictions
    plot_pred_target_values(pred)
    ##
    ## the model (2 features) can be represented as
    ## h(x) = w0 + w1x1 + w2x2
    ## h(x) = 0 - @ desision boundary
    ## 0 = w0 + w1x1 + w2x2
    ## equation for x2: x2 = -(w0 + w1x1)/w2
    ##
    ## store the model parameters
    t0 = model.intercept_[0]
    t1 = model.coef_[0][0]
    t2 = model.coef_[0][1]
    ##
    ## create a range of X_1 values
    x1_bound = np.linspace(-1, 1, 100)
    ## determine the corresponding X_2 values
    x2_bound = -(t0 + (x1_bound * t1)) / t2
    ## plot the desision boundary
    plt.plot(x1_bound, x2_bound, '--', c="red", label='Desision Boundary')
    ##
    plt.title(f'Linear SVM (C={penalty}, 2 features) with desision boundary')
    plt.xlabel('X_1 features')
    plt.ylabel('X_2 features')
    plt.legend()


## (c) (i)
##
## square features
X1_2 = X1**2
X2_2 = X2**2
X_2 = np.column_stack((X1,X2,X1_2,X2_2))
##
## create logistic regression model using sklearn and fit the (4 feature) data
model = LogisticRegression(penalty='none',solver='lbfgs').fit(X_2,y)
print(f'Model - Logistic Regression (4 features)')
print(f'Penalty - none')
print(f'Intercept - {model.intercept_}')
print(f'Slope - {model.coef_}\n')

## (c) (ii)
##
plt.figure(6)
## plot training data
plot_train_target_values(y)
## predict linear regression model (4 features) target values
pred = model.predict(X_2)
## plot model predictions
plot_pred_target_values(pred)

## (c) (iii)
##
## h(x) = w0 + w1x1 + w2x2 + w3x1^2 + w4x2^2
## h(x) = 0 - @ desision boundary
## w0 + w1x1 + w2x2 + w3x1^2 + w4x2^2 = 0
## w4x2^2 + w2x2 + (w0 + w1x1 + w3x1^2) = 0
## can now obtain equation using quadratic roots formula
## equation for x2: x2 = (-(w2) +||- np.sqrt((w2**2) - (4 * w4 * (w0 + (t1 * x1) + (w3 * (x1**2)))))) / (2 * w4)
##
## store the model parameters
t0 = model.intercept_[0]
t1 = model.coef_[0][0]
t2 = model.coef_[0][1]
t3 = model.coef_[0][2]
t4 = model.coef_[0][3]
##
## create a range of X_1 values
x1_bound = np.linspace(-1, 1, 100)
## determine the corresponding X_2 values
x2_bound = (-(t2) + np.sqrt((t2**2) - (4 * t4 * (t0 + (t1 * x1_bound) + (t3 * (x1_bound**2)))))) / (2 * t4)
## plot the desision boundary
plt.plot(x1_bound, x2_bound, '--', c="red", label='Desision Boundary')
##
##
plt.title('Logistic Regression (4 features) with desision boundary')
plt.xlabel('X_1 features')
plt.ylabel('X_2 features')
plt.legend()
plt.figure(1)
plt.title('Training Data (2 features)')
plt.xlabel('X_1 features')
plt.ylabel('X_2 features')
plt.legend()
plt.figure(2)
plt.title('Logistic Regression (2 features) with decision boundary')
plt.xlabel('X_1 features')
plt.ylabel('X_2 features')
plt.legend()

## display plots
plt.show()