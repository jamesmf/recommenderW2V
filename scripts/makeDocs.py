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
import cPickle

def main():
    STD_SCALAR  = 10
    "begin"
    userStuff()  
    print "done with userStuff"                            
    ratings     = getRatings() 
    print "done with averages"                         
    makeDocuments("../data/raw/",ratings,STD_SCALAR)    #Writes a document for each user


"""takes in a line from ratings and returns what to write in that user's doc"""    
def writeDoc(fields,ratings,STD_SCALAR):
    movie           = str(int(fields[1]))
    user_rating     = float(fields[2])
    value           = ratings[movie]
    mean_rating     = float(value[0])
    std_rating      = float(value[1])+.00000001
    z               = (user_rating-mean_rating)/std_rating
    writeNum        = z*STD_SCALAR

    #This returns the word to write and how many times to write it in a tuple
    if z > 0:
        out = ("L_"+movie+" ",int(np.floor(writeNum)))
    elif z < 0:
        out = ("D_"+movie+" ",int(np.floor(-writeNum)))
    else:
        out = ("",0)
    return out
    
"""iterates over all the ratings and writes user's documents"""    
def makeDocuments(inFolder,ratings,STD_SCALAR):

    files   = listdir(inFolder+"train/")

    for fi in files:
        count = 0
        with open(inFolder+"train/"+fi,'rb') as lines:
            for line in lines:
                count   +=1
                if (count%1000) == 0:
                    print count
                fields  = line.split("::")
                if len(fields) > 1:
                    with open(inFolder+"../processed/user_"+fields[0]+".txt",'a') as f2:
                        out = writeDoc(fields,ratings,STD_SCALAR)
                        for i in range(0,out[1]):
                            f2.write(out[0])   

"""either creates rating dictionary or reads it from a flat txt file"""            
def getRatings():
    if not isfile("../data/out/avgRatings.txt"):
        with open("../data/raw/train/TrainMe.txt",'rb') as f:
            avgRatings  = ratingsAvg(f)
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

"""
gets the mean and stdev rating for each movie
returns a dict of "movie": "[mean,stdev]" 
"""
def ratingsAvg(data):
    movies  = {}
    #dsp = data.split("\n")
    for rating in data:
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

"""gets the number of ratings given and the average rating for each user"""
def userStuff():
    with open("../data/raw/train/TrainMe.txt",'rb') as f1:
        
        userStats   = {}
        count   = 0
        for rating in f1:
            tsp = rating.split("::")
            if len(tsp) > 1:
                if userStats.has_key(tsp[0]):
                    userStats[tsp[0]].append(float(tsp[2]))
                else:
                    userStats[tsp[0]]  = [float(tsp[2])]
        out = {}            
        ratTots     = []
        for k,v in userStats.iteritems():
            ratTots.append(len(v))

        numMean     = np.mean(np.array(ratTots))
        numSTD      = np.std(np.array(ratTots))

        for k,v in userStats.iteritems():
            norm    = (len(v) - numMean)/numSTD
            out[k]  = str(np.mean(v))+"\t"+str(norm)
            
         

                 
         
        with open("../data/out/users.pickle",'wb') as f:
            cp = cPickle.Pickler(f)
            cp.dump(out)
    
if __name__ == "__main__":
    main()