# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:51:36 2015

@author: jmf
"""

import numpy as np
from os import listdir
from os.path import isfile
#import scipy.stats as statz

def ratingsAvg(data):
    movies  = {}
    dsp = data.split("\n")
    for rating in dsp:
        tsp = rating.split("\t")
        if len(tsp) > 1:
            if movies.has_key(tsp[1]):
                movies[tsp[1]].append(int(tsp[2]))
            else:
                movies[tsp[1]]  = [int(tsp[2])]
    out ={}            
    for k,v in movies.iteritems():
        out[k]  = np.mean(v)
        
    return out
    
def getUsers():
    us  = {}
    di  = "../data/processed/"
    l   = listdir(di)
    for fi in l:
        with open(di+fi,'rb') as f:
            us[fi]  = f.read()
    return us
    
def getTestSet(testLoc):
    with open(testLoc,'rb') as f:
        return f.read().split("\n")
    
    
    
def getRatings():
    if not isfile("../data/out/avgRatings.txt"):
        with open("../data/raw/train/ua.base",'rb') as f:
            avgRatings  = ratingsAvg(f.read())
        with open("../data/out/avgRatings.txt",'wb') as f2:
            for movie in avgRatings:        
                f2.write(str(movie)+"\t"+str(avgRatings[movie])+"\n")
    
    with open("../data/out/avgRatings.txt",'rb') as f:
        x   = f.read().split("\n")
    out = {}
    for line in x:
        sp  = line.split("\t")
        if len(sp) > 1:
            out[sp[0]]  = sp[1]
   
    return out
    
def findNearest(userNum):
    with open("../data/out/nearest.txt",'rb') as f:
        lines   = f.read().split("\n")
        userNum = int(userNum) - 1
        neigh   = lines[userNum]
    return neigh.split("|")[1:]
    

    
def predict(test,users,ratings):
    tot = 0
    for line in test:
        sp  = line.split("\t")
        if len(sp) > 1:
            user= sp[0]
            mv  = sp[1]
            rtg = sp[2]
            ns  = findNearest(user)
            e   = check(ns,mv,rtg,users,ratings)
            tot +=e
    print tot/len(test[:-1])

def check(ns,mv,rtg,users,ratings):
    out = [0,0,0,0,0]
    for n in ns:
        interpret(mv,users[n],out)
    if np.sum(out) > 10:
        #guess   = np.argmax(out)+1
        guess   = rate2guess(out)
    else:
        if ratings.has_key(mv):
            guess   = ratings[mv]
        else:
            guess   = 3.06
    print out, guess, rtg, (int(rtg)-float(guess))**2
    RMSE    = (int(rtg)-float(guess))**2
    return RMSE  

def rate2guess(vec):
    s   = sum(vec)
    o   = [vec[i-1]*i for i in range(1,6)]
    return sum(o)*1./s     
    
def interpret(num,line,out):
    line    = line.split(" ")
    like    = "like_"+str(num)
    dislike = "dislike_"+str(num)
    neutral = "neutral_"+str(num)
   
    if line.count(like) == 3:
        out[4]+=1
    if line.count(like) == 1:
        out[3]+=1
    if line.count(neutral) > 0:
        out[2]+=1
    if line.count(dislike) == 1:
        out[1]+=1
    if line.count(dislike) == 3:
        out[0]+=1

    
    

           
           
           
def main():
    ratings = getRatings()
    users   = getUsers()
    testSet = getTestSet("../data/raw/test/ua.test")
    pred    = predict(testSet,users,ratings)            
            
if __name__ == "__main__":
    main()