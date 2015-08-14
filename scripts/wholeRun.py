# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:42:16 2015

This controller runs through an entire instance from doc creation to eval

@author: jmf
"""

import  sys
import  os
from os      import listdir
from os.path import isdir
from os      import mkdir


def readConfigFile():
    if len(sys.argv) > 1:
        config  = sys.argv[1]
    with open(config,'rb') as f:
        x   = f.read().split("\n")
        folder  = x[0]
        topics  = x[1]
        alpha   = x[2]
        beta    = x[3] 
        if not isdir(folder):
            mkdir(folder)
        
        
def main():
    
