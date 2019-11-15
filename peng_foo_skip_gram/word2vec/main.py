from train import *
from extra_preprocessing import *
from gensim.models.keyedvectors import KeyedVectors


# run processed data file. This will take in the json file prepared, generate corpus generate dictionarys for onehot encoding and the embedding.
# then you can run the train.py file.

model = KeyedVectors.load_word2vec_format('./model_test.txt', binary=False)

word = "universitylife"
print("the hastag #" + word + " is most similar to:")
similarWords = model.most_similar(positive=word)
for i in range (0, 5):
    print("#" + similarWords[i][0])

model2 = KeyedVectors.load_word2vec_format('../data/word_vectors_dict.json', binary=False)

word = "universitylife"
print("the hastag #" + word + " is most similar to:")
similarWords = model2.most_similar(positive=word)
for i in range (0, 5):
    print("#" + similarWords[i][0])

