from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU

import numpy as np 
from sklearn import preprocessing
import cPickle

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
def ratingsToData(ratings,vectors,userInfo,movieAvg):
    X   = []
    y   = []
    for line in ratings:
        #r   = [0,0,0,0,0]
        sp  = line.split("::")
        if len(sp) > 1:
            user= sp[0]
            mv  = sp[1]
            rtg = sp[2]
            userword    = "user_"+user.strip()+".txt"
            likeword    = "like_"+mv
            dislword    = "dislike_"+mv

            if vectors.has_key(userword):
                uservec     = vectors[userword]
            else:
                uservec     = np.zeros(20)
            if vectors.has_key(likeword):
                likevec     = vectors[likeword]
            else:
                likevec     = np.zeros(20)
            if vectors.has_key(dislword):
                disvec      = vectors[dislword]
            else:
                disvec      = np.zeros(20)
                
            avgRating   = userInfo[user][0]
            numRated    = userInfo[user][0]
            movieMean   = movieAvg[mv][0]
            movieStd    = movieAvg[mv][1]

                
            example     = np.append(movieMean,movieStd)
            example     = np.append(example,numRated)
            example     = np.append(example,avgRating)
            example     = np.append(example,uservec)
            example     = np.append(example,likevec)
            example     = np.append(example,disvec)

            X.append(example)
            #print example, len(example)
            #stop = raw_input("stop")
            y.append(rtg)
    return X, y

def main():
    np.random.seed(1000)
    
    #####
    data_folder = "../data/"
    out_folder = "../data/out/"
    batch_size = 4
    nb_epoch = 10
    
    ### load train and test ###
    testRatings = getTestSet("../data/raw/test/ra.test")
    trainRatings= getTrainSet("../data/raw/train/ra.train")
    vectors     = loadThemVectors()
    userInfo    = getUserInfo()
    movieAvg    = getMovieAvg()
    X,y         = ratingsToData(trainRatings,vectors,userInfo,movieAvg)
    testX,testy = ratingsToData(testRatings,vectors,userInfo,movieAvg)
    
    size        = len(X[0])
    
    
    #train  = pd.read_csv(data_folder+'train.csv', index_col=0)
    #test  = pd.read_csv(data_folder+'test.csv', index_col=0)
    print "Data Read complete"
#    
#    Y = train.Survived
#    train.drop('Survived', axis=1, inplace=True)
#    
#    columns = train.columns
#    test_ind = test.index
#    
#    train['Age'] = train['Age'].fillna(train['Age'].mean())
#    test['Age'] = test['Age'].fillna(test['Age'].mean())
#    train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
#    test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
#    
#    category_index = [0,1,2,4,5,6,8,9]
#    for i in category_index:
#        print str(i)+" : "+columns[i]
#        train[columns[i]] = train[columns[i]].fillna('missing')
#        test[columns[i]] = test[columns[i]].fillna('missing')
#    
#    train = np.array(train)
#    test = np.array(test)
       
    ### label encode the categorical variables ###
#    for i in category_index:
#        print str(i)+" : "+str(columns[i])
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(train[:,i]) + list(test[:,i]))
#        train[:,i] = lbl.transform(train[:,i])
#        test[:,i] = lbl.transform(test[:,i])
    
    ### making data as numpy float ###
#    train = train.astype(np.float32)
#    test = test.astype(np.float32)
    #Y = np.array(Y).astype(np.int32)
    
    model = Sequential()
    model.add(Dense(size, 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, 100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    
    
    model.add(Dense(512, 1))
    model.add(Activation('tanh'))
    
    model.compile(loss='mean_square_error', optimizer='rmsprop')
    model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.20)
    preds = model.predict(testX,batch_size=batch_size)
    
    #pred_arr = []
    for i in range(0,preds):
        print preds[i], testy[i]
    
    ### Output Results ###

if __name__ == "__main__":
    main()