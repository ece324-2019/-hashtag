import os
import json
from ekphrasis.classes.segmenter import Segmenter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.preprocessing


seg = None
glove = None

def tokenizor_hashtags(hashtags_in_post): # takes in a list of hashtags, and return a list of list of words(each hashtag is broken into a list of string)
    seg = Segmenter(corpus="twitter")
    rtl = []
    for item in hashtags_in_post:
        rtl.append(seg.segment(item))
    return rtl

# load the glovefile
def load_glove_model(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model
# this generate a vector by combining wordvectors in the hashtag
def generate_vector_from_hashtags(glove, hashtag):
    vec_list = []
    vec_rtv = np.zeros((100),dtype=float)
    rtl = seg.segment(hashtag).split()
    error = True
    for item in rtl:
        try:
            vec_list.append(glove[item.lower()])
            error = False
        except KeyError:
            k = 3
    if error:
        rtv = np.random.rand(100)
        rtv = (rtv - np.mean(rtv))/np.std(rtv)
        return rtv

    for i in vec_list:
        vec_rtv = vec_rtv + i
    vec_rtv = vec_rtv/len(vec_list)
    return vec_rtv

def get_one_hot_from_list(corpus):
    #corpus = ["instagood", "ThrowBackThursday", "forThegram", "GodIsGreat", "CheeseParole"]
    vectorizer = CountVectorizer()
    dict_of_corpus_words = vectorizer.fit_transform(corpus) # this creates a dictionary that corresponds hashtags to numbers that are not onehot encoded
    dict_of_corpus_words = vectorizer.vocabulary_
    dict_of_corpus_words_keys = list(dict_of_corpus_words.keys()) # get the keys
    le = sklearn.preprocessing.LabelEncoder().fit(dict_of_corpus_words_keys) # do the label creation and onehot encoding
    integery_data = le.transform(list(dict_of_corpus_words.keys()))
    ohe = sklearn.preprocessing.OneHotEncoder().fit(integery_data.reshape((-1,1)))
    # print(dict_of_corpus_words)
    onehot_data = ohe.transform(integery_data.reshape((-1,1)))
    oneHotDict = {}
    for i in range(0, len(dict_of_corpus_words_keys)):
       oneHotDict[dict_of_corpus_words_keys[i]] = onehot_data[i].todense().tolist()
    # print(oneHotDict)
    return oneHotDict

# generate the dictionaries needed
def get_Dictionaries():
    data = None
    hashtagData = []
    with open('../../crawler/output.json') as json_data:
        data = json.load(json_data)
        json_data.close()
    for post in data:
        try:
            hashtagData.append(post["hashtags"])
        except KeyError:
            hashtagData.append([])
    flat_list = [item for sublist in hashtagData for item in sublist]
    # one_hot_dict = get_one_hot_from_list(flat_list)
    one_hot_dict = {}
    with open('../data/word2id.json') as json_data:
        one_hot_dict = json.load(json_data)
        json_data.close()
    word_vectors_dict = {}
    for item in list(one_hot_dict.keys()):
        word_vectors_dict[item.lower()] = generate_vector_from_hashtags(glove, item).tolist()
    with open('../data/one_hot_dict.json', 'w') as fp:
        json.dump(one_hot_dict, fp)
    with open('../data/word_vectors_dict.json', 'w') as fp:
        json.dump(word_vectors_dict, fp)

if __name__ == '__main__':
    seg = Segmenter(corpus="twitter")
    glove = load_glove_model("../data/glove.6B.100d.txt")
    hashtags = []
    with open('../../crawler/output.json') as json_data:
        data = json.load(json_data)
        json_data.close()
        for post in data:
            try:
                hashtags.append(post["hashtags"])
            except KeyError:
                hashtags.append([])
    file = open("../data/hashtag_corpus.txt", "w")  # write mode
    for post in hashtags:
        if post != []:
            line = ""
            for tag in post:
                line = line + tag + " "
            file.write(line)
            file.write("\n")
    get_Dictionaries()
    file.close()
