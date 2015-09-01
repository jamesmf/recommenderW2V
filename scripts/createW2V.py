# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:21:40 2015

take user documents and create a word2vec doc out of it


@author: frickjm
"""
from random import shuffle
import numpy as np
from os import listdir


WINDOWSIZE  = 20
NUMLINES    = 20


def getRandom(user,words,outfile):
    global WINDOWSIZE, NUMLINES
    numLines    = min(max(NUMLINES,len(words)/10),400)
    for x in range(numLines):
        shuffle(words)
        words2  = words
        [words2.insert(pos,user) for pos in range(len(words)/WINDOWSIZE + 1)]
        new     = ' '.join(words2)
        with open(outfile,'a') as of:
            of.write(new+'\n')

def main():

    OUTFILE     = "../data/W2V_movies.txt"
    folder      = "../data/processed/"
    x           = [folder+a for a in listdir(folder)]
    user        = [u[u.rfind("/")+1:u.rfind(".")] for u in x]
    
    shuffle(x)    
    count   = 0
    for fileName in x:
        with open(fileName,'rb') as f:
            user    = fileName[fileName.rfind("/")+1:]
            words   = f.read().split(" ")     
            getRandom(user,words,OUTFILE)
            count+=1
            if (count%1000) == 0:
                print count,"/",len(x)
            
#
#    for fileName in x:
#        with open(fileName,'rb') as f:
#            user    = fileName[fileName.rfind("/")+1:]
#            words   = f.read().split(" ")     
#            getRandom(user,words,NUMLINES,NUMWORDS,OUTFILE)

    
if __name__ == "__main__":
    main()