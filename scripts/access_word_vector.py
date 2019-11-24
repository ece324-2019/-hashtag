from gensim.models.keyedvectors import KeyedVectors
import torch
import numpy as np
# run processed data file. This will take in the json file prepared, generate corpus generate dictionarys for onehot encoding and the embedding.
# then you can run the train.py file.


if __name__ == '__main__':
    model0 = KeyedVectors.load_word2vec_format('./Data/vectors/205wordVectors.txt', binary=False)
    model1 = KeyedVectors.load_word2vec_format('./Data/vectors/405wordVectors.txt', binary=False)
    model2 = KeyedVectors.load_word2vec_format('./Data/vectors/1003wordVectors.txt', binary=False)
    model3 = KeyedVectors.load_word2vec_format('./Data/wordVectors.txt', binary=False)
    word = "interiorphotographer"

    similarWords = model0.most_similar(positive=word)
    print("In the 20 embedding dimension model the hastag #" + word + " is most similar to:")
    for i in range (0, 5):
        print("#" + similarWords[i][0])
    similarWords = model1.most_similar(positive=word)
    print("In the 40 embedding dimension model the hastag #" + word + " is most similar to:")
    for i in range(0, 5):
        print("#" + similarWords[i][0])
    similarWords = model2.most_similar(positive=word)
    print("In the 100 embedding dimension model the hastag #" + word + " is most similar to:")
    for i in range(0, 5):
        print("#" + similarWords[i][0])
    similarWords = model3.most_similar(positive=word)
    print("In the chosen model the hastag #" + word + " is most similar to:")
    for i in range(0, 5):
        print("#" + similarWords[i][0])


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