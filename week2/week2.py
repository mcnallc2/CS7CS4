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

## (a) (i)
plt.figure(1)
for i, val in enumerate(y):
    if val == 1:
        plt.scatter(X[i][0], X[i][1], s=60, color='blue', marker='+')
    elif val == -1:
        plt.scatter(X[i][0], X[i][1], s=30, color='black', marker='o')

## (a) (ii)
model = LogisticRegression(penalty='none',solver='lbfgs').fit(X,y)
print(f'intercept {model.intercept_[0]}, slope_1 {model.coef_[0][0]}, slope_2 {model.coef_[0][1]}')

pred = model.predict(X)
for i, val in enumerate(pred):
    if val == 1:
        plt.scatter(X[i][0], X[i][1], s=25, color='green', marker='+')
    elif val == -1:
        plt.scatter(X[i][0], X[i][1], s=5, color='yellow', marker='o')

## h(x) = w0 + w1x1 + w2x2
## h(x) = 0 - @ desision boundary
## 0 = w0 + w1x1 + w2x2
## equation for x2: x2 = -(w0 + w1x1)/w2
x1_bound = np.linspace(-1, 1, 100)
x2_bound = -(model.intercept_[0] + (x1_bound * model.coef_[0][0])) / model.coef_[0][1]
plt.plot(x1_bound, x2_bound, '--', c="red")


plt.figure(2)
for i, val in enumerate(y):
    if val == 1:
        plt.scatter(X[i][0], X[i][1], s=60, color='blue', marker='+')
    elif val == -1:
        plt.scatter(X[i][0], X[i][1], s=30, color='black', marker='o')


for penalty in [0.001, 1, 1000]:

    model = LinearSVC(C=penalty, max_iter=90000).fit(X,y)
    print(f'penalty - {penalty}')
    print(f'intercept - {model.intercept_[0]}')
    print(f'slope_1 - {model.coef_[0][0]}')
    print(f'slope_2 - {model.coef_[0][1]}\n')

    pred = model.predict(X)
    for i, val in enumerate(pred):
        if val == 1:
            plt.scatter(X[i][0], X[i][1], s=25, color='green', marker='+')
        elif val == -1:
            plt.scatter(X[i][0], X[i][1], s=5, color='yellow', marker='o')

    ## h(x) = w0 + w1x1 + w2x2
    ## h(x) = 0 - @ desision boundary
    ## 0 = w0 + w1x1 + w2x2
    ## equation for x2: x2 = -(w0 + w1x1)/w2
    x1_bound = np.linspace(-1, 1, 100)
    x2_bound = -(model.intercept_[0] + (x1_bound * model.coef_[0][0])) / model.coef_[0][1]
    plt.plot(x1_bound, x2_bound, '--', c="red")



plt.figure(1)
# plt.title('')
plt.xlabel('X1 features')
plt.ylabel('X2 features')
plt.legend(['desision boundary'])

## display plots
plt.show()