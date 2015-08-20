# recommenderW2V
embedding users and items in the same space using a Word2Vec model 

First attempt (no parameter tweaking) RMSE on MovieLens10M: 0.83

The pipeline is:
  
  1 - Separate MovieLens10M ratings.dat by user (splitData.py)
  
      - 80% of each user's ratings are in the Training set, 20% are in the Test set
      - Some movies in the Test set aren't in the Training set - this harms the model, but in order to keep the 80/20 split random, this isn't addressed.  The model simply has no guarantee of information about each item
      - Output: 
        data/raw/train/TrainMe.txt 
        data/raw/test/TestMe.txt  


  2 - Convert the training dataset into user documents (makeDocs.py)
  
      - Normalize by movie (item)
      - For each rating (of the form user|movie|rating) determine the rating's z-score
        - if z > 0:
          - write "like_"+itemID in the user's document
        - else:
          - write "dislike_"+itemID in the user's document
      - The number of times the word is written is proportional to the z-score (this scalar is a parameter of the model)
      - Output:
        - data/processed/*.txt

  3 - Create a single file on which to train the Word2Vec model (createW2V.py)
  
      - For each user, write a number of lines out depending on how many ratings the user has
      - For each line:
        - Shuffle the words in the user's document
        - Write out the first N words in the list (N is a parameter)
        - Write the userID in the middle of the "sentence"

  4 - Train the Word2Vec model on the resulting file (word2vec/makeVectors.sh and word2vec/bin2txt.py)
  
      - Parameters - all the parameters of the W2V model
      - Output:
        - data/vectors.txt

  5 - Train the Neural Net using Keras:
  
      - Parameters - the configuration of NN
      - Create training examples from:
          - userVector
          - likeMovieVector
          - dislikeMovieVector
          - userAvgRating
          - userNumMoviesRated
          - movieAvgRating
          - movieStdRating
      - Read in the test set and make a guess for each rating
          - If the model hasn't seen the item before, the like/dislike vectors are zeros
