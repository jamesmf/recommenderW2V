# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:32:58 2015

Splits the ratings.dat file into a training and test set

@author: frickjm
"""
from random import shuffle
import numpy as np

def main():
    movieTestSet    = []
    movieTrainSet   = []
    with open("../data/raw/ratings.dat",'rb') as f:
        ratings     = f.read().split("\n")
        
    userStats   = {}
    for line in ratings:
        tsp = line.split("::")
        if len(tsp) > 1:
            if userStats.has_key(tsp[0]):
                userStats[tsp[0]].append(line)
            else:
                userStats[tsp[0]]  = [line]

    for k,v in userStats.iteritems():
        shuffle(v)
        l   = len(v)
        te  = int(np.round(l*0.2))
        tr  = l - te
        train   = v[:-te]
        test    = v[-te:]
        with open("../data/raw/train/TrainMe.txt",'a') as f1:
            for x in train:
                f1.write(x+"\n")
                movieTrainSet.append(x.split("::")[1])
                
        with open("../data/raw/test/TestMe.txt",'a') as f2:
            for y in test:
                f2.write(y+"\n")
                movieTestSet.append(y.split("::")[1])
                
    trSet   = set(movieTrainSet)
    teSet   = set(movieTestSet)
    
    print "len trainset",len(trSet)
    print "len testset", len(teSet)
    
    for movie in teSet:
        if not movie in trSet:
            print movie
            stop=raw_input("dangit!")

if __name__ == "__main__":
    main()    