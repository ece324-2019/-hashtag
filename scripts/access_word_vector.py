from gensim.models.keyedvectors import KeyedVectors
import torch
import numpy as np
# run processed data file. This will take in the json file prepared, generate corpus generate dictionarys for onehot encoding and the embedding.
# then you can run the train.py file.




def returnListOfEmbedding(listOfHashtag):
    model = KeyedVectors.load_word2vec_format('../peng_foo_skip_gram/word2vec/model_test.txt', binary=False)

    rtv = []
    shape=(300,)
    for item in listOfHashtag:
        try:
            rtv.append(model.wv[item])
        except:
            rtv.append(np.zeros(shape=shape))
    return rtv

#print(returnListOfEmbedding(test))