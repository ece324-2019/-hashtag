from gensim.models.keyedvectors import KeyedVectors


# run processed data file. This will take in the json file prepared, generate corpus generate dictionarys for onehot encoding and the embedding.
# then you can run the train.py file.

model = KeyedVectors.load_word2vec_format('../peng_foo_skip_gram/word2vec/model_test.txt', binary=False)

word = "universitylife"
print("the hastag #" + word + " is most similar to:")

similarWords = model.most_similar(positive=word)
similarWords = model.similar_by_vector(vector=similarWords[i][1]) # this allow you find similar words/vector from vector

for i in range (0, 5):
    print("#" + similarWords[i][0])


