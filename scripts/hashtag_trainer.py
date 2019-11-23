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

def evaluate(output, label):
    print(output.size())
    print(label.size())

class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)
        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            # print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 10000 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
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
            # if i % 100 == 0:
                # print(loss)
                # print("step: %d, Loss: %0.8f, lr: %0.6f" % (i, loss.item(), self.optimizer.param_groups[0]['lr']))
                # evaluate()
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

def train_vectors():
    w2v = Word2Vec(log_filename="./Data/hashtag_corpus.txt", output_filename="./Data/wordVectors.txt", embedding_dimension = 40, iteration=7)
    w2v.train()

    
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
    train_vectors()
    generate_dict_of_hashtag()
# call this to run all the functions related to hashtag training
def training():
    gen_corpus()
    train_vectors()
    generate_dict_of_hashtag()
