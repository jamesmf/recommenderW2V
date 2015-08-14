# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:51:39 2015

@author: jmf

Creates documents for each user by:

1) obtaining the mean rating for each movie
2) writes a term in the user's document "like_movieID" or "dislike_movieID"
    - if rating > mean, write like
    - if rating < mean, write dislike
    - write term proportional to stdev above/below mean 
"""

import sys
from os import listdir
from os.path import isfile
import numpy as np

def main():
    #folder  = sys.argv[1]
    #ratings = [3,1,1,1,3]
    STD_SCALAR  = 10
    ratings     = getRatings()
    makeDocuments("../data/raw/",ratings,STD_SCALAR)
    
def writeDoc(fields,ratings,STD_SCALAR):
    movie           = str(int(fields[1]))
    user_rating     = float(fields[2])
    value           = ratings[movie]
    mean_rating     = float(value[0])
    std_rating      = float(value[1])+.00000001
    z               = (user_rating-mean_rating)/std_rating
    writeNum        = z*STD_SCALAR
    #print user_rating, mean_rating, std_rating, z, writeNum
    if z > 0:
        out = ("like_"+movie+" ",int(np.floor(writeNum)))
    elif z < 0:
        out = ("dislike_"+movie+" ",int(np.floor(-writeNum)))
    else:
        out = ("",0)
    return out
    
    
def makeDocuments(inFolder,ratings,STD_SCALAR):

    files   = listdir(inFolder+"train/")
    for fi in files:
        with open(inFolder+"train/"+fi,'rb') as f:
            lines   = f.read().split("\n")
        for line in lines:
            fields  = line.split("::")
            if len(fields) > 1:
                with open(inFolder+"../processed/user_"+fields[0]+".txt",'a') as f2:
                    out = writeDoc(fields,ratings,STD_SCALAR)
                    for i in range(0,out[1]):
                        f2.write(out[0])   
                        
def item2user():
    files   = listdir("../data/processed/")
    for a in files:
        with open("../data/processed/"+a,'rb') as f:
            content = f.read()
        b   = a.replace("item","user")
        with open("../data/processed/"+b,'wb') as f2:
            f2.write(content)
            
def getRatings():
    if not isfile("../data/out/avgRatings.txt"):
        with open("../data/raw/train/ra.train",'rb') as f:
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
            out[sp[0]]  = [sp[1], sp[2]]
 
    return out

def ratingsAvg(data):
    movies  = {}
    dsp = data.split("\n")
    for rating in dsp:
        tsp = rating.split("::")
        if len(tsp) > 1:
            if movies.has_key(tsp[1]):
                movies[tsp[1]].append(float(tsp[2]))
            else:
                movies[tsp[1]]  = [float(tsp[2])]
    out ={}            
    for k,v in movies.iteritems():
        out[k]  = str(np.mean(v))+"\t"+str(np.std(v))
        
    return out


    
if __name__ == "__main__":
    main()