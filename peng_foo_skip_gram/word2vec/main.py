from train import *
from extra_preprocessing import *
from gensim.models.keyedvectors import KeyedVectors


# run processed data file. This will take in the json file prepared, generate corpus generate dictionarys for onehot encoding and the embedding.
# then you can run the train.py file.


model = KeyedVectors.load_word2vec_format('./model_test.txt', binary=False)
model_b = KeyedVectors.load_word2vec_format('./model_test_baseline.txt', binary=False)

word = "sundaycheatmeal"


similarWords = model_b.most_similar(positive=word)
print("In the baseline model the hastag #" + word + " is most similar to:")
for i in range (0, 5):
    print("#" + similarWords[i][0])


print("\n==========================================\n")

similarWords = model.most_similar(positive=word)
print("In the actual model the hastag #" + word + " is most similar to:")
for i in range (0, 5):
    print("#" + similarWords[i][0])