#!/usr/bin/python

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


## read and format data in csv file
df = pd.read_csv("week1.csv",comment='#',header=None)
X = np.array(df.iloc[:,0]); X = X.reshape(-1,1)
y = np.array(df.iloc[:,1]); y = y.reshape(-1,1)


## normailse X-values data
mean_x = np.mean(X)
# sig = np.std(data)
sig_x = max(X) - min(X)
x_norm = ((X - mean_x) / sig_x)


## function to perform gradient descent
def gradient_descent(X, y, lr, iterations):
    ##
    ## init epoch counter
    ## init theta values as 0
    ## init prev mean-squared-error value
    ## init empty array for logging info
    epoch_count = 0
    theta0 = 0
    theta1 = 0
    mse_prev = 0
    theta0_log = []
    theta1_log = []
    mse_log = []
    ##
    ## gradient decent algorithm - minimise the cost function of specified iterations
    for _ in range(iterations):
        y_hat = theta0 + (theta1 * x_norm)
        ##
        ## find partial derivative & multiply by learning rate
        ## then update the theta values
        theta0 += ((2 * -lr) / len(y)) * sum(y_hat - y)
        theta1 += ((2 * -lr) / len(y)) * sum((y_hat - y) * x_norm)
        ##
        ## find new mean-squared-error
        mse = ((1 / len(y)) * sum((y_hat - y)**2))
        ##
        ## if the change in mse to prev iteration is very small then finish
        if abs(mse_prev - mse) < 0.000000001:
            break
        ##
        ## set mse prev to curr
        mse_prev = mse
        ##
        ## log new theta values and mse values
        theta0_log.append(float(theta0))
        theta1_log.append(float(theta1))
        mse_log.append(float(mse))
        ##
        ## increment epoch counter
        epoch_count += 1
        ##
    ##
    ## return the final theta values and error, the theta logs, the mse logs and the number epochs performed
    return (theta0, theta1, theta0_log, theta1_log, mse, mse_log, epoch_count)


## create 3D figure
plt.figure(3)
ax = plt.gca(projection='3d')

## (b) (i)
## create array of learning rates
learning_rates = [0.001, 0.01, 0.1, 1]
##
## perfrom gradient decent for each learning rate
for lr in learning_rates:
    ##
    ## call gradient decent function and return theta values, final error and all logs
    (theta0, theta1, theta0_log, theta1_log, mse, mse_log, epoch_count) = gradient_descent(x_norm, y, lr, 1000)
    print(f'Learning Rate = {lr}')
    print(f'theta_0 - {theta0}')
    print(f'theta_1 - {theta1}')
    print(f'final error - {mse}\n')
    ##
    ## get linear regression model using minimised error theta values
    y_hat = theta0 + (theta1 * x_norm)
    ##
    ## plot the linear regression model with training data
    plt.figure(1)
    plt.plot(X, y_hat, linewidth=2)
    ##
    ## plot the reduction in mean-squared-error after each iteration
    plt.figure(2)
    plt.plot(range(epoch_count), mse_log, linewidth=2)
    ##
    ## plt the 3D reduction in mse on cost function contour map
    plt.figure(3)
    ax.plot(theta0_log, theta1_log, mse_log)


## plots
## linear regression models and training data
plt.figure(1)
plt.scatter(X, y, s=10, color='black')
plt.title('Maunal Linear Regression models using muliple learning rates')
plt.xlabel('X values')
plt.ylabel('y values')
plt.legend(['lr = 0.001', 'lr = 0.01', 'lr = 0.1', 'lr = 1', 'training data'])
##
## reduction in error over time
plt.figure(2)
plt.title('Cost function vs epochs with multiple learning rates')
plt.xlabel('epochs')
plt.ylabel('J(theta0, theta1)')
plt.legend(['lr = 0.001', 'lr = 0.01', 'lr = 0.1', 'lr = 1'])


## create a mesh grid of theta values for the 3D contour
t0 = np.linspace(0, 2000, 100)
t1 = np.linspace(0, 2000, 100)
T0, T1 = np.meshgrid(t0, t1)
##
## init cost function 2D array
cost_function = np.zeros((100,100))
##
## determine the cost function values for the theta values
for i in range(100):
    for j in range(100):
        y_hat = t0[i] + (t1[j] * x_norm)
        cost_function[i][j] = ((1 / len(y)) * sum((y_hat - y)**2))
    ##
##
## create 3D figure
plt.figure(3)
ax = plt.gca(projection='3d')
##
## plot the cost function as a contour map
ax.contour3D(T0, T1, cost_function, 30, cmap='binary')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J(theta0, theta1)')
plt.title('3D representation of Cost Function Minimisation')
plt.legend(['lr = 0.001', 'lr = 0.01', 'lr = 0.1', 'lr = 1'])


## (b) (ii)
## baseline model
base_value = np.min(y) + ((np.max(y) - np.min(y))/2)
print(f'Baseline model y-value - {base_value}')
## find baseline mean-squared-error
baseline_mse = ((1 / len(y)) * sum((base_value - y)**2))
print(f'Baseline cost function value - {baseline_mse}')


## (b) (iii)
## linear regression using sklearn
model = LinearRegression().fit(X,y)
##
y_pred = (X * model.coef_) + model.intercept_
## plotting predictions
plt.figure(4)
plt.scatter(X, y, s=10, color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.title('Linear regression model using SKLEARN')
plt.xlabel('X values')
plt.ylabel('y values')
plt.legend(['predictions','training data'])


## display plots
plt.show()
