import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_reader import DataReader, Word2vecDataset
import numpy as np
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import sys
# from word2vec.data_handler import DataHanlder
from hashtag_data_util import DataHanlder, gen_corpus
import json
from matplotlib import pyplot as plt
from gensim.models.keyedvectors import KeyedVectors

def evaluate(pos_score, neg_score):
    corr_pos = (pos_score).squeeze() > 0.5
    corr_neg = (neg_score).squeeze() > 0.5

    tot_positive = (corr_pos.size()[0])
    true_positive = (corr_pos.sum().item())
    tot_negative = corr_neg.size()[0] * corr_neg.size()[1]
    true_negative = corr_neg.sum().item()

    accuracy = (true_positive + true_negative)/(tot_positive + tot_negative)
    sensitivity = true_positive/(true_positive + tot_negative - true_negative + 0.0000001)
    f1 = 2*true_positive/(2*true_positive + tot_positive - true_positive + tot_negative - true_negative+ 0.0000001)
    return {"accuracy":accuracy, "sensitivity":sensitivity, "f1":f1}



class Word2Vec:
    def __init__(self, log_filename: str,
                 output_filename: str,
                 embedding_dimension: int=100,
                 batch_size: int=128,
                 iteration: int=1,
                 initial_lr: float=0.025,
                 min_count: int=1,
                 sub_sampling_t: float = 1e-5,
                 neg_sampling_t: float = 0.75,
                 neg_sample_count: int = 5,
                 half_window_size: int = 5,
                 read_data_method: str='memory',
                 seed = 1):
        """
        init func

        """
        torch.manual_seed(seed)
        self.data = DataHanlder(log_filename=log_filename,
                                batch_size=batch_size,
                                min_count=min_count,
                                sub_sampling_t=sub_sampling_t,
                                neg_sampling_t=neg_sampling_t,
                                neg_sample_count=neg_sample_count,
                                half_window_size=half_window_size,
                                read_data_method=read_data_method,
                                seed=seed)
        self.output_filename = output_filename
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.half_window_size = half_window_size
        self.iter = iteration
        self.initial_lr = initial_lr
        # weight_tensor = self.get_pretrained_weight_tensor()
        # self.sg_model = SkipGramModel(len(self.data.vocab), self.embedding_dimension, init_weights = weight_tensor)
        self.sg_model = SkipGramModel(len(self.data.vocab), self.embedding_dimension, init_weights=None, seed=seed)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.sg_model.cuda()
        self.optimizer = optim.SGD(self.sg_model.parameters(), lr=self.initial_lr)

    # this load the pretrained vector made up using the words
    def get_pretrained_weight_tensor(self):
        id2word = None
        word_vectors = None
        with open('./Data/id2word.json') as json_data:
            id2word = json.load(json_data)
            json_data.close()
        with open('./Data/word_vectors_dict.json') as json_data:
            word_vectors = json.load(json_data)
            json_data.close()
        keys = sorted(id2word.keys())
        list_of_weigths = []
        print(len(word_vectors.keys()))
        print(len(id2word.keys()))
        for key in keys:
            list_of_weigths.append(word_vectors[id2word[key].lower()])
        list_of_weigths = torch.FloatTensor(list_of_weigths)
        return list_of_weigths


    def train(self):
        i = 0
        # total 2 * self.half_window_size * self.data.total_word_count,
        # for each sent, (1 + 2 + .. + half_window_size) * 2 more pairs has been calculated, over all * sent_len
        # CAUTION: IT IS NOT AN ACCURATE NUMBER, JUST APPROXIMATELY COUNT.
        approx_pair = 2 * self.half_window_size * self.data.total_word_count - \
                      (1 + self.half_window_size) * self.half_window_size * self.data.sentence_len
        batch_count = self.iter * approx_pair / self.batch_size
        lossList = []
        iList = []
        accuracy_i_list = []
        accuracy_list = []
        sensitivity_list = []
        f1_list = []
        for pos_u, pos_v, neg_samples in self.data.gen_batch():
            i += 1
            if self.data.sentence_cursor > self.data.sentence_len * self.iter:
                # reach max iter
                break
            # train iter
            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_samples))
            if self.use_cuda:
                pos_u, pos_v, neg_v = [i.cuda() for i in (pos_u, pos_v, neg_v)]
            # print(len(pos_u), len(pos_v), len(neg_v))
            self.optimizer.zero_grad()
            loss_and_posVal_and_negVal = self.sg_model.forward(pos_u, pos_v, neg_v)
            loss = loss_and_posVal_and_negVal[0]
            loss.backward()
            self.optimizer.step()
            lossList.append(loss.item())
            iList.append(i)
            if i % 100 == 0:
                # print(loss)
                # print("step: %d, Loss: %0.8f, lr: %0.6f" % (i, loss.item(), self.optimizer.param_groups[0]['lr']))
                pos_score = loss_and_posVal_and_negVal[1]
                neg_score = loss_and_posVal_and_negVal[2]
                accuracy_i_list.append(i)
                temp_out = evaluate(pos_score, neg_score)
                accuracy_list.append(temp_out["accuracy"])
                f1_list.append(temp_out["f1"])
                sensitivity_list.append(temp_out["sensitivity"])
            if i % (100000 // self.batch_size) == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.sg_model.save_embedding(self.data.id2word, self.output_filename, self.use_cuda)
        lossList = np.array(lossList)
        accuracy_list = np.array(accuracy_list)
        accuracy_i_list = np.array(accuracy_i_list)
        f1_list = np.array(f1_list)
        iList = np.array(iList)

        # plt.title("Loss & Accuracy when training word2Vec")
        # plt.subplot(3, 1, 1)
        # plt.plot(iList, lossList, label="loss")
        # plt.ylabel("loss")
        # plt.subplot(3, 1, 2)
        # plt.plot(accuracy_i_list, accuracy_list, label="accuracy")
        # plt.ylabel("accuracy")
        # plt.subplot(3, 1, 3)
        # plt.plot(accuracy_i_list, f1_list, label="f1")
        # plt.ylabel("f1 score")
        # plt.xlabel("batch")
        # plt.legend()
        # plt.show()
        return [iList, lossList, accuracy_i_list, accuracy_list, f1_list]

def train_vectors():
    w2v = Word2Vec(log_filename="./Data/hashtag_corpus.txt", output_filename="./Data/wordVectors.txt", embedding_dimension = 100, iteration=10)
    w2v.train()
def hyper_search():
    embedding_size = [20, 40, 100, 300]
    window_size = [3, 5, 10, 20]
    embedding_size = [20]
    window_size = [3]
    out_path = "./Data/vectors/"
    index = 1
    for emb_size in embedding_size:
        for window in window_size:
            w2v = Word2Vec(log_filename="./Data/hashtag_corpus.txt", output_filename=out_path+str(emb_size)+str(window)+"wordVectors.txt", embedding_dimension = emb_size, iteration=20, half_window_size=window)
            [iList, lossList, accuracy_i_list, accuracy_list, f1_list] = w2v.train()
            plt.subplot(len(window_size), len(embedding_size), index)
            plt.plot(accuracy_i_list, f1_list, label="f1")
            plt.ylabel("f1")
            index = index + 1
    plt.plot


def generate_dict_of_hashtag():
    listOfHashtag = None
    with open('./Data/word2id.json') as json_file:
        data = json.load(json_file)
        listOfHashtag = data.keys()
    dict_tags = {}
    model = KeyedVectors.load_word2vec_format('./Data/wordVectors.txt', binary=False)
    for item in listOfHashtag:
        dict_tags[item] = model.wv[item].tolist()
    with open('./Data/hashtag_vector_dict.json', 'w') as outfile:
        json.dump(dict_tags, outfile)
    return dict_tags
if __name__ == '__main__':
    gen_corpus()
    hyper_search()
    generate_dict_of_hashtag()
# call this to run all the functions related to hashtag training
def training():
    gen_corpus()
    train_vectors()
    generate_dict_of_hashtag()
