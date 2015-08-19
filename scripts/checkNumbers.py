# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:01:14 2015

@author: frickjm
"""

import numpy as np

with open("../data/out/predictions.csv",'rb') as f:
    lines   = f.read().split("\n")
 
num     = len(lines) 
print num
 
squareErrorSum  = 0
roundedSQErroerSum = 0

for line in lines:
    cells   = line.split("\t")
    if len(cells) > 1:        
        guess   = np.float(cells[0].replace("[",'').replace("]",''))
        ans     = np.float(cells[1])                
        squareErrorSum += (guess-ans)**2

    
meanSquareError  = squareErrorSum*1./num



RMSE  = np.sqrt(meanSquareError)


print "RMSE", RMSE

