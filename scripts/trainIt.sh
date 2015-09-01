make
./word2vec -train ../data/W2V_movies.txt -output ../data/out/vectors.bin -cbow 1 -size 50 -window 20 -negative 10 -hs 0 -sample 1e-5 -threads 40 -binary 1 -iter 5 -min-count 1
python vec2txt.py ../data/out/vectors.bin ../data/out/vectors.txt
