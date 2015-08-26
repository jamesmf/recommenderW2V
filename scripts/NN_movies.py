from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU

import numpy as np 
from sklearn import preprocessing
import cPickle

"""DEFINE SOME CONSTANTS"""
AVG_NUM_RATED   = 20
AVG_RATING_VAL  = 3.19
AVG_STD_VAL     = 0.93
VECTOR_LENGTH = 25


"""Returns the word2vec vectors as a dictionary of "term": np.array()"""
def loadThemVectors():
    outMat  = {}
    with open("../data/vectors.txt",'rb') as f:
        for line in f.readlines():
            sp  = line.split()
            name= sp[0].strip()            
            arr     = np.array(sp[1].split(","))
            arr     = [float(x) for x in arr]
            outMat[name]    = arr           
    return outMat

"""returns a list of the lines in the Test ratings file"""    
def getTestSet(testLoc):
    with open(testLoc,'rb') as f:
        return f.read().split("\n")
        
"""returns a list of the lines in the Train ratings file"""        
def getTrainSet(trainLoc):
    with open(trainLoc,'rb') as f:
        return f.read().split("\n")
        
"""get user information created in makeDocs.py"""
def getUserInfo():
    with open("../data/out/users.pickle",'rb') as f:
        return cPickle.load(f)
        
"""get average ratings for each movie, created in makeDocs.py"""
def getMovieAvg():
    with open("../data/out/avgRatings.txt",'rb') as f:
        x   = f.read().split("\n")
    out = {}
    for line in x:
        sp  = line.split("\t")
        if len(sp) > 1:
            out[sp[0]]  = [sp[1], sp[2]]
    return out
        
        
"""takes the ratings lines and the w2v vectors and produces for each line
one example to be fed to a neural net"""
def ratingsToData(ratings,vectors,userInfo,movieAvg,subset,AVG_RATING_VAL=3.19,AVG_STD_VAL=0.93,AVG_NUM_RATED=20):
    global VECTOR_LENGTH
    X   = []
    y   = []
    rLen    =  len(ratings)
    if not (subset == -1):
        ratings = ratings[subset[0]:subset[1]]
    for line in ratings:
        #r   = [0,0,0,0,0]
        sp  = line.split("::")
        if len(sp) > 1:
            user= sp[0]
            mv  = sp[1]
            rtg = float(sp[2])
            userword    = "user_"+user.strip()+".txt"
            likeword    = "like_"+mv
            dislword    = "dislike_"+mv

            if vectors.has_key(userword):
                uservec     = vectors[userword]
            else:
                uservec     = np.zeros(VECTOR_LENGTH)
                
            if vectors.has_key(likeword):
                likevec     = vectors[likeword]
            else:
                likevec     = np.zeros(VECTOR_LENGTH)

            if vectors.has_key(dislword):
                disvec      = vectors[dislword]
            else:
                disvec      = np.zeros(VECTOR_LENGTH)
                
            if userInfo.has_key(user):
                avgRating   = float(userInfo[user].split("\t")[0])
                numRated    = float(userInfo[user].split("\t")[1])
            else:
                print "NO USER"
                avgRating   = AVG_RATING_VAL
                numRated    = AVG_NUM_RATED

            if movieAvg.has_key(mv):
                movieMean   = float(movieAvg[mv][0])
                movieStd    = float(movieAvg[mv][1])
            else:
                print "NO MOVIE!!!"
                movieMean   = avgRating
                movieStd    = AVG_STD_VAL
                
            example     = np.append(movieMean,movieStd)
            #example     = np.append(example,numRated)
            example     = np.append(example,avgRating)
            example     = np.append(example,uservec)
            example     = np.append(example,likevec)
            example     = np.append(example,disvec)


            X.append(example)
            #print example, len(example)
            #stop = raw_input("stop")
            y.append(rtg)

    return np.array(X), np.array(y)
    
def getTrainSeq(ratings,vectors,userInfo,movieAvg,batchSize,block):
    ind     = block*batchSize
    subset  = (ind, ind+batchSize)
    Xbatch, ybatch  = ratingsToData(ratings,vectors,userInfo,movieAvg,subset)
    return np.array(Xbatch), np.array(ybatch)
    
def getTrainBatch(ratings,vectors,userInfo,movieAvg,batchSize):
    ind     = int(np.floor(np.random.rand()*(len(ratings) - batchSize)))
    subset  = (ind, ind+batchSize)
    Xbatch, ybatch  = ratingsToData(ratings,vectors,userInfo,movieAvg,subset)
    return np.array(Xbatch), np.array(ybatch)

def getTestBatch(ratings,vectors,userInfo,movieAvg,batchSize,block):
    ind     = block*batchSize
    subset  = (ind, ind+batchSize)
    Xbatch, ybatch  = ratingsToData(ratings,vectors,userInfo,movieAvg,subset)
    return np.array(Xbatch), np.array(ybatch)
    
def makeModel(numNeurons,descriptions):
    model   = Sequential()
    for layerNum in range(1,len(numNeurons)):
        prevLayerSize   = numNeurons[layerNum-1]
        layerSize       = numNeurons[layerNum]
        model.add(Dense(prevLayerSize,layerSize))        
        
        if descriptions[layerNum].find("relu") > -1:
            model.add(Activation('relu'))
            
        if descriptions[layerNum].find("dropout") > -1:
            model.add(Dropout(0.5))
        
    return model
        

def main():
    np.random.seed(1000)

    
    #####
    data_folder = "../data/"
    out_folder  = "../data/out/"
    batchSize   = 32
    numberEpochs= 150000
    
    ### load training data and other relevant data ###
    trainRatings= getTrainSet("../data/raw/train/TrainMe.txt")
    vectors     = loadThemVectors()
    userInfo    = getUserInfo()
    movieAvg    = getMovieAvg()
    
    firstBatchX, firstBatchy   = getTrainBatch(trainRatings,vectors,userInfo,movieAvg,batchSize)
    size        = len(firstBatchX[0])
    del firstBatchX, firstBatchy
    
    print size
    print "Data Read complete"
   
    """define the Neural Net"""
    numNeurons      = [size,200,100,1]    #number of units in each layer
    descriptions    = ['input', 'relu', 'relu dropout','output'] #'type' of each layer, as parsed in makeModel() 
    
    model           = makeModel(numNeurons,descriptions)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    
    
  
    
    """One pass through to ensure that every datapoint has been seen once"""    
    for i in range(0,int(np.ceil(len(trainRatings)*1./batchSize))):
        batchX,batchy   = getTestBatch(trainRatings,vectors,userInfo,movieAvg,batchSize,i)
        loss            = model.train(batchX,batchy)
        if (i%10000) == 0:
            print str(i)+"/"+str(numberEpochs)
            
    """Get batches of data and train on them"""
    for i in range(0,numberEpochs):
        batchX,batchy   = getTrainBatch(trainRatings,vectors,userInfo,movieAvg,batchSize)
        loss            = model.train(batchX, batchy)
        if (i%1000) == 0:
            print str(i)+"/"+str(numberEpochs)  

    
    del trainRatings #we've finished training, now we load the test set
    
    """define X and y as the test set"""
    testRatings = getTestSet("../data/raw/test/TestMe.txt")
    meanError   = 0
    sqError     = 0
    batchSize   = 10000    
    """test the model in batches"""

    for i in range(0,int(np.ceil(len(testRatings)*1./batchSize))):    
        batchX,batchy   = getTestBatch(testRatings,vectors,userInfo,movieAvg,batchSize,i)
        preds   = model.predict(batchX,batch_size=batchSize)
    
        ### Write out results ###
        with open("../data/out/predictions.csv",'a') as f:
            for i in range(0,len(preds)):
                #print preds[i], batchy[i]
                sqError   += np.float(preds[i]-batchy[i])**2
                f.write(str(preds[i])+"\t"+str(batchy[i])+"\n")
    
    meanError   = sqError/len(testRatings)
    RMSE        = np.sqrt(meanError)
    print RMSE
    with open("../data/out/predictions.csv",'a') as f:
        f.write(str(RMSE))
        
if __name__ == "__main__":
    main()