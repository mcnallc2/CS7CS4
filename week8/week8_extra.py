#!/usr/bin/python

import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt

## specify input and kernal dimensions
N = 200
K = 3
M = int((K-1)/2)

## determine edge pixels based on kernel size
edge = []
for e in range(M):
    if not e==M:
        edge.append(e)
    edge.append(N-1-e)

## function to perform convolution
def convolution(_input, _kernal):
    ## init convolution output
    output=[]
    ## iterate through pixel rows
    for row in range(len(_input)):
        ## init output rows
        output_row=[]
        ## iterate through pixel in each row
        for col in range(len(_input[row])):
            ## init conv mask
            mask = 0
            ## if pixel is not on the edge of the image (Valid Padding)
            if not ((row in edge) or (col in edge)):
                ## interate through each kernel element 
                for i in range(len(_kernal)):
                    for j in range(len(_kernal[i])):
                        ## sum each kernel*pixel element
                        mask += (_input[row-M+i][col-M+j] * _kernal[i][j])
                    ##
                ## append to output
                output_row.append(mask)
            ##
        ## if row is not at the edge
        if output_row:
            ## append row to output array
            output.append(output_row)
        ##
    return output

## specify kernals
kernal1 = [[-1, -1, -1],
           [-1,  8, -1],
           [-1, -1, -1]]

kernal2 = [[ 0, -1,  0],
           [-1,  8, -1],
           [ 0, -1,  0]]

## open image
im = Image.open('whitetiger.jpg')
rgb = np.array(im.convert('RGB'))
r=rgb[:,:,0] # array of R pixels

## get outputs for each kernal
output1 = convolution(r, kernal1)
output2 = convolution(r, kernal2)

## show images
Image.fromarray(np.uint8(r)).show()
Image.fromarray(np.uint8(output1)).show()
Image.fromarray(np.uint8(output2)).show()

## plotting accuracy vs l1 reg weights
weights = ['0', '0.0001', '0.001', '0.01', '0.1', '1']
TA = [64, 63, 58, 45, 37, 22]
VA = [50, 50, 49, 41, 35, 21]

plt.figure(1)
plt.plot(weights, TA, label='Training Accuracy')
plt.plot(weights, VA, label='Validation Accuracy')
plt.title('Prediction Accuracy vs L1 reg weight (5K data points)')
plt.xlabel('L1 weight')
plt.ylabel('Accuracy')
plt.legend()
plt.show()