import numpy as np
# from word2vec.model import SkipGramModel
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import sys
# from word2vec.data_handler import DataHanlder
from data_handler import DataHanlder
import json
from matplotlib import pyplot as plt

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
                 read_data_method: str='memory'):
        """
        init func

        """
        self.data = DataHanlder(log_filename=log_filename,
                                batch_size=batch_size,
                                min_count=min_count,
                                sub_sampling_t=sub_sampling_t,
                                neg_sampling_t=neg_sampling_t,
                                neg_sample_count=neg_sample_count,
                                half_window_size=half_window_size,
                                read_data_method=read_data_method)
        self.output_filename = output_filename
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.half_window_size = half_window_size
        self.iter = iteration
        self.initial_lr = initial_lr
        weight_tensor = self.get_pretrained_weight_tensor()
        self.sg_model = SkipGramModel(len(self.data.vocab), self.embedding_dimension, init_weights = weight_tensor)
        # self.sg_model = SkipGramModel(len(self.data.vocab), self.embedding_dimension, init_weights=None)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.sg_model.cuda()
        self.optimizer = optim.SGD(self.sg_model.parameters(), lr=self.initial_lr)

    # this load the pretrained vector made up using the words
    def get_pretrained_weight_tensor(self):
        id2word = None
        word_vectors = None
        with open('../data/id2word.json') as json_data:
            id2word = json.load(json_data)
            json_data.close()
        with open('../data/word_vectors_dict.json') as json_data:
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
            loss = self.sg_model.forward(pos_u, pos_v, neg_v)
            print(loss)
            loss.backward()
            self.optimizer.step()
            lossList.append(loss.item())
            iList.append(i)
            if i % 100 == 0:
                # print(loss)
                print("step: %d, Loss: %0.8f, lr: %0.6f" % (i, loss.item(), self.optimizer.param_groups[0]['lr']))
            if i % (100000 // self.batch_size) == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.sg_model.save_embedding(self.data.id2word, self.output_filename, self.use_cuda)
        lossList = np.array(lossList)
        iList = np.array(iList)
        plt.title("Loss when training word2Vec")
        plt.plot(iList, lossList, label="training")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.show()
if __name__ == '__main__':
    w2v = Word2Vec('../data/hashtag_corpus.txt', 'model_test.txt', iteration=10)
    w2v.train()

