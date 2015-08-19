# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:41:53 2015

@author: frickjm
"""



import codecs
from gensim.models import Word2Vec
import json
 
def export_to_file(path_to_model, output_file):
	output = codecs.open(output_file, 'w' , 'utf-8')
	model = Word2Vec.load_word2vec_format(path_to_model, binary=True)
	vocab = model.vocab
	for mid in vocab:
		#print(model[mid])
		print(mid)
		vector = list()
		for dimension in model[mid]:
			vector.append(str(dimension))
		#line = { "mid": mid, "vector": vector  }
		vector_str = ",".join(vector)
		line = mid + "\t"  + vector_str
		#line = json.dumps(line)
		output.write(line + "\n")
	output.close()
 
 
def main():
     export_to_file("../data/out/vectors.bin","../data/vectors.txt")
     
if __name__ == "__main__":
    main()