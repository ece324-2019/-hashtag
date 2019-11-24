import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

# credits to https://github.com/Andras7/word2vec-pytorch/blob/master/model.py



class SkipGramModel(nn.Module):
    """
    Skip-Gram model
    """
    def __init__(self, vocab_size: int, emb_dimension: int=200, init_weights=None, seed=1):
        torch.manual_seed(seed)
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.init_emb()
        if init_weights is not None:
            self.u_embeddings.weight.data = init_weights
    def init_emb(self):
        """
        init the weight as original word2vec do.

        :return: None
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """
        forward process.
        the pos_u and pos_v shall be the same size.
        the neg_v shall be {negative_sampling_count} * size_of_pos_u
        eg:
        5 sample per batch with 200d word embedding and 6 times neg sampling.
        pos_u 5 * 200
        pos_v 5 * 200
        neg_v 5 * 6 * 200

        :param pos_u:  positive pairs u, list
        :param pos_v:  positive pairs v, list
        :param neg_v:  negative pairs v, list
        :return:
        """
        emb_u = self.u_embeddings(pos_u)  # batch_size * emb_size
        emb_v = self.v_embeddings(pos_v)  # batch_size * emb_size
        emb_neg = self.v_embeddings(neg_v)  # batch_size * neg sample size * emb_size
        # print(pos_u)
        # A[1]
        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        alt_pos_score = F.sigmoid(pos_score)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        alt_neg_score = F.sigmoid(-neg_score)
        neg_score = F.logsigmoid(-neg_score)
        return [-1 * (torch.sum(pos_score) + torch.sum(neg_score)), alt_pos_score, alt_neg_score]

    def save_embedding(self, id2word: dict, file_name: str='./Data/word_vectors.txt', use_cuda: bool=False):
        """
        Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.

        :param id2word: map from word id to word.
        :param file_name: file name.
        :param use_cuda:
        :return:
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


class Baseline(nn.Module):

    def __init__(self, in_size):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(in_size, 100);
        self.fc2 = nn.Linear(100, 1);

    def forward(self, x, lengths=None):
        x=self.fc1(x);
        x=self.fc2(x);

	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return x


class CNN(nn.Module):
    def __init__(self, kernel_num = 10, fc1_num = 100, output_dim=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, kernel_num, 2) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, 4)
        self.conv4 = nn.Conv2d(kernel_num, kernel_num, 5)
        self.bn2 = nn.BatchNorm1d(10 * 9 * 9)
        self.bn3 = nn.BatchNorm1d(fc1_num)
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 9 * 9, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, output_dim) # output Layer
    def forward(self, input):
        x = (F.leaky_relu(self.maxpool(self.conv1(input))))
        x = (F.leaky_relu(self.maxpool(self.conv2(x))))
        x = (F.leaky_relu(self.maxpool(self.conv3(x))))
        x = (F.leaky_relu(self.maxpool(self.conv4(x))))
        x = self.bn2(x.view(-1, 10 * 9 * 9))
        x = self.bn3((F.leaky_relu(self.fc1(x))))
        x = self.fc2(x)
        return x
