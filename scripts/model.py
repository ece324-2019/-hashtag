import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
from torch.nn import init
import numpy as np

# credits to https://github.com/Andras7/word2vec-pytorch/blob/master/model.py



class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, embedding_vectors):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True) # this has to be changed
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True) # this has to be changed

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

        val = np.random.rand(100,)
        val = torch.FloatTensor(val)
        self.u_embeddings.weight[:, 0] = val
        print(self.u_embeddings.weight[:, 0])

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

<<<<<<< HEAD

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
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 9 * 9, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, output_dim) # output Layer
    def forward(self, input):
        x = (F.relu(self.maxpool(self.conv1(input))))
        x = (F.relu(self.maxpool(self.conv2(x))))
        x = (F.relu(self.maxpool(self.conv3(x))))
        x = (F.relu(self.maxpool(self.conv4(x))))
        x = x.view(-1, 10 * 9 * 9)
        x = (F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
<<<<<<< HEAD
        return x
>>>>>>> a290fd7ccf2cd7ce4ebb491f6d906a86a85d3896
=======
        return x
>>>>>>> a290fd7ccf2cd7ce4ebb491f6d906a86a85d3896
