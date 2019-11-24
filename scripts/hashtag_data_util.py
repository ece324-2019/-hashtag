import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.preprocessing
from collections import defaultdict
from collections import deque


seg = None
glove = None

np.random.seed(1)


class DataHanlder:
    """
    input date handler
    for storing/sampling data, etc.
    """
    def __init__(self, log_filename: str,
                 min_count: int=5,
                 sub_sampling_t: float=1e-5,
                 neg_sampling_t: float=0.75,
                 batch_size: int=64,
                 neg_sample_count: int=5,
                 half_window_size: int=2,
                 read_data_method='memory',
                 seed = 1):
        np.random.seed(seed)
        """
        init function.

        :param log_filename: word2vec format -> item1 item2 ... itemk\n, each line a user log
        :param min_count: the same min_count as in word2vec
        :param sub_sampling_t: sub sampling rate, higher than that would be sub sampled using
                                the word2vec paper using:    p(w_i) = 1 - sqrt(sub_sampling / freq)
                                the word2vec code using:     p(w_i) = 1 - (sqrt(sub_sampling / freq) + sub_sampling / freq)
                                we use word2vec code subsampling method here.
        :param neg_sampling_t: the negative sampling t as in the word2vec. seems not shown in the paper, but implemented in
                                the code:
                                    p(w_i) = f(w_i) ** neg_sampling / sum(f(w_i) ** neg_sampling for w_i in vocab)
        :param read_data_method: method to read data:
                                    'memory': load all the sentence to memory, fast but cost memory.
                                    'file': load data from file, slower but save memory
        """
        assert read_data_method in ('memory', 'file')
        self.log_filename = log_filename
        self.min_count = min_count
        self.sub_sampling_t = sub_sampling_t
        self.neg_sampling_t = neg_sampling_t
        self.sentences = deque()
        self.read_data_method = read_data_method
        print('read dataset...')
        self.vocab, self.word2id, self.id2word, self.total_word_count, self.sentence_len = self.gen_vocab()
        with open('./Data/id2word.json', 'w') as fp:
            json.dump(self.id2word, fp)
        with open('./Data/word2id.json', 'w') as fp:
            json.dump(self.word2id, fp)
        print(f'got vocab {len(self.vocab)}, total_word_count {self.total_word_count}')
        print('gen negative sample table...')
        self.neg_sample_table = self.gen_negative_sample_table()
        print('done.\ngen sub sampling table...')
        self.sub_sampling_table = self.gen_subsample_table()
        print('done.')
        self.batch_size = batch_size
        self.sentence_cursor = 0  # sentence cursor for generating batch
        self.neg_sample_count = neg_sample_count
        self.half_window_size = half_window_size

    def gen_vocab(self):
        """
        from log file generate vocabulary
        :return: {item_id: freq}
        """
        assert self.log_filename != ''
        vocab_freq_dict = defaultdict(int)
        total_word_count = 0
        total_sent_count = 0
        with open(self.log_filename, encoding='utf-8') as f:
            for line in f:
                total_sent_count += 1
                item_ids = line.strip().split()
                if self.read_data_method == 'memory':
                    self.sentences.append(item_ids)
                for item_id in item_ids:
                    vocab_freq_dict[item_id] += 1
                    total_word_count += 1
        vocab, word2id, id2word = {}, {}, {}
        index = 0
        for item_id, freq in vocab_freq_dict.items():
            if freq < self.min_count:
                continue
            vocab[item_id] = freq
            word2id[item_id] = index
            id2word[index] = item_id
            index += 1
        return vocab, word2id, id2word, total_word_count, total_sent_count

    def gen_subsample_table(self):
        """
        sub sampling rate, higher than that would be sub sampled using
            the word2vec paper using:    p(w_i) = 1 - sqrt(sub_sampling / freq)
            the word2vec code using:     p(w_i) = 1 - (sqrt(sub_sampling / freq) + sub_sampling / freq)
        we use word2vec code sub sampling method here.
        :return: {word_id: sample_score}
        """
        def sub_sampling(_freq):
            return (self.sub_sampling_t / 1.0 / _freq) ** 0.5 + self.sub_sampling_t / 1.0 / _freq
        # word freq count to word freq ratio
        sub_sample_tbl = {item: freq / 1.0 / self.total_word_count
                          for item, freq in self.vocab.items()
                          if freq / 1.0 / self.total_word_count > self.sub_sampling_t}
        # freq to score
        sub_sample_tbl = {item: sub_sampling(_freq) for item, _freq in sub_sample_tbl.items()}
        # word to id
        sub_sample_tbl = {self.word2id[i]: j for i, j in sub_sample_tbl.items() if j < 1}
        return sub_sample_tbl

    def gen_negative_sample_table(self):
        """
        implemented same as word2vec c code.
        The way this selection is implemented in the C code is interesting. They have a large array with 100M elements
        (which they refer to as the unigram table). They fill this table with the index of each word in the vocabulary
        multiple times, and the number of times a word’s index appears in the table is given by P(wi) * table_size.

            p(w_i) = f(w_i) ** neg_sampling / sum(f(w_i) ** neg_sampling for w_i in vocab)

        :return:
        """
        sample_tbl_size = 1e8
        sample_tbl = []
        pow_freq = np.array(list(self.vocab.values())) ** self.neg_sampling_t
        pow_total_freq = sum(pow_freq)
        r = pow_freq / pow_total_freq
        count = np.round(r * sample_tbl_size)
        for item_id, _count in enumerate(count):
            sample_tbl += [item_id] * int(_count)
        sample_tbl = np.array(sample_tbl)
        return sample_tbl

    def gen_batch(self):
        """
        yield batch
        :return: pos_pairs -> [(w1, w2) * batch_size ], neg samples -> [self.neg_sample_count * batch_size]
        """
        f = open(self.log_filename)
        pos_pairs = []
        pos_1, pos_2, neg_samples = [], [], []
        while True:
            while len(pos_pairs) < self.batch_size:
                # pos
                sentence = []
                if self.read_data_method == 'memory':
                    sentence = self.sentences.popleft()
                elif self.read_data_method == 'file':
                    sentence = f.readline()
                    if not sentence:
                        f = open(self.log_filename)
                        sentence = f.readline()
                    sentence = sentence.strip().split()
                self.sentence_cursor += 1
                # to word id
                word_ids = [self.word2id[item_id] for item_id in sentence if item_id in self.word2id]
                for i, word_id in enumerate(word_ids):
                    pos_pairs += [(word_id, word_ids[j])
                                  for j in range(max(0, i - self.half_window_size),
                                                 min(i + self.half_window_size + 1, len(word_ids) - 1))
                                  if j != i]
                if self.read_data_method == 'memory':
                    self.sentences.append(sentence)
            # neg
            for pos1, pos2 in pos_pairs[len(neg_samples):]:
                pos_1.append(pos1)
                pos_2.append(pos2)
                neg_samples.append(self.negative_sampling(pos1, pos2))
            yield (pos_1[:self.batch_size], pos_2[:self.batch_size], neg_samples[:self.batch_size])
            pos_pairs = pos_pairs[self.batch_size:]
            pos_1 = pos_1[self.batch_size:]
            pos_2 = pos_2[self.batch_size:]
            neg_samples = neg_samples[self.batch_size:]

    def negative_sampling(self, pos1, pos2):
        """
        negative sample, shall not equal to pos1 and pos2
        :param pos1:
        :param pos2:
        :return:
        """
        negs = []
        while len(negs) < self.neg_sample_count:
            _negs = np.random.choice(self.neg_sample_table, size=self.neg_sample_count - len(negs))
            negs += [i for i in _negs if i != pos1 and i != pos2]

        return negs[:self.neg_sample_count]

    def thread_read_data_(self):
        """
        useless since GIL exists.
        though it is a pure IO thread, it still does not accelerate.
        :return:
        """
        f = open(self.log_filename)
        while True:
            if len(self.sentences) < 1000:
                sentence = f.readline()
                if not sentence:
                    f = open(self.log_filename)
                    sentence = f.readline()
                item_ids = sentence.strip().split(' ')
                self.sentences.append(item_ids)


def test():
    handler = DataHanlder('../data/trainset.txt', read_data_method='memory')
    i = 0
    import time
    start = time.time()
    for pos_1, pos_2, neg_samples in handler.gen_batch():
        if handler.sentence_cursor >= 2 * handler.total_word_count:
            break
        i += 1
        if i > 1000:
            break
    end = time.time()
    print('1000 iter using', end - start)

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
def get_Dictionaries(hashtagData):
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
def gen_corpus():
    # seg = Segmenter(corpus="twitter")
    # glove = load_glove_model("../data/glove.6B.100d.txt")
    hashtags = []
    with open('./images.json') as json_data:
        data = json.load(json_data)
        json_data.close()
        for post in data.keys():
            try:
                hashtags.append(data[post])
            except KeyError:
                hashtags.append([])
    file = open("./Data/hashtag_corpus.txt", "w")  # write mode
    for post in hashtags:
        if post != []:
            line = ""
            for tag in post:
                line = line + tag + " "
            try:
                file.write(line)
                file.write("\n")
            except:
                pass
    # get_Dictionaries(hashtags)
    file.close()
if __name__ == '__main__':
    gen_corpus()
    # seg = Segmenter(corpus="twitter")
    # glove = load_glove_model("../data/glove.6B.100d.txt")
    hashtags = []
    with open('./images.json') as json_data:
        data = json.load(json_data)
        json_data.close()
        for post in data.keys():
            try:
                hashtags.append(data[post])
            except KeyError:
                hashtags.append([])
    file = open("./Data/hashtag_corpus.txt", "w")  # write mode
    for post in hashtags:
        if post != []:
            line = ""
            for tag in post:
                line = line + tag + " "
            file.write(line)
            file.write("\n")
    # get_Dictionaries(hashtags)
    file.close()
