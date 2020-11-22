#!/usr/bin/python

import numpy as np
from PIL import  Image

## specify input and kernal dimensions
N = 200
K = 3
M = int((K-1)/2)
edge = []
for e in range(M):
    if not e==M:
        edge.append(e)
    edge.append(N-1-e)

## function to perform convolution
def convolution(_input, _kernal):
    ##
    output=[]
    for row in range(len(_input)):
        output_row=[]
        for col in range(len(_input[row])):
            mask = 0
            if not ((row in edge) or (col in edge)):
                for i in range(len(_kernal)):
                    for j in range(len(_kernal[i])):
                        mask += (_input[row-M+i][col-M+j] * _kernal[i][j])
                    ##
                output_row.append(mask)
            ##
        if output_row:
            output.append(output_row)
        ##
    return output

## specify kernals
_kernal1 = [[-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]]

_kernal2 = [[ 0, -1,  0],
            [-1,  8, -1],
            [ 0, -1,  0]]


im = Image.open('whitetiger.jpg')
rgb = np.array(im.convert('RGB'))
r=rgb[:,:,0] # array of R pixels

output1 = convolution(r, _kernal1)
output2 = convolution(r, _kernal2)

Image.fromarray(np.uint8(r)).show()
Image.fromarray(np.uint8(output1)).show()
Image.fromarray(np.uint8(output2)).show()

