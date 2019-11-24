#----Contains all the codes for training----#
import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pyplot
from model import *
from dataset import *
from access_word_vector import *
import hashtag_trainer as ht
import json
from gensim.models.keyedvectors import KeyedVectors

class train:
    def __init__(self,cnn_out_dimention,data,loss_function='MSELoss',model='baseline',lr=0.001,epochs=100):
        self.out_dimention=cnn_out_dimention
        if loss_function=="MSELoss":
            self.loss_fnc = nn.MSELoss()
        if loss_function=="CrossEntropy":
            self.loss_fnc = nn.functional.binary_cross_entropy_with_logits()
        if loss_function=="cos":
            self.loss_fuc = nn.CosineEmbeddingLoss()
        print(nn.CosineEmbeddingLoss())
        self.model=CNN(output_dim=self.out_dimention)
        self.epochs=epochs
        self.data=data
        self.model_name=model
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []
        self.optimizer=optim.Adam(self.model.parameters(), lr=lr)
        self.hashtag_to_embedding_dict = ht.generate_dict_of_hashtag()

        self.gensim = KeyedVectors.load_word2vec_format('./Data/wordVectors.txt', binary=False)
        with open('./Data/id2word.json') as json_data:
            self.id_to_word = json.load(json_data)
            json_data.close()
        with open('./Data/word2id.json') as json_data:
            self.word_to_id = json.load(json_data)
            json_data.close()
    def measure_acc(self,outputs, labels):
        acc=0
        for i in range(0, len(outputs)):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            beta=1
            output=outputs[i].detach().numpy()
            label=labels[i].squeeze().detach().numpy()
            acc_temp=0
            for j in range(len(output)):
                if (output[j]>0.5 and label[j]>0.5):
                    tp+=1
                elif(output[j]<=0.5 and label[j]<=0.5):
                    tn+=1
                elif(output[j]>=0.5 and label[j]<=0.5):
                    fp+=1
                elif(output[j]<=0.5 and label[j]>=0.5):
                    fn+=1
            # acc_temp=(1+beta*beta)*tp/((1+beta*beta)*tp+beta*beta*fn+fp + 0.00000001)
            acc_temp=(tp+tn)/(tp+tn+fn+fp)
            acc+=acc_temp
        return acc/len(outputs)
    def compare_with_embeddings(self,outputs,hashtags):
        hashtags_dic=self.hashtag_to_embedding_dict
        embeddings=hashtags_dic.items()
        embeddings=np.asarray(embeddings)
        embeddings=torch.from_numpy(embeddings)
        outputs_copy=torch.zeros([len(outputs),len(hashtags)])
        for i in range(len(outputs)):
            for j in range(len(hashtags)):
                try:
                    outputs_copy[i,j]=torch.dot(outputs[i],embeddings[j])/torch.norm(outputs[i])/torch.norm(embeddings[j])
                except:
                    outputs_copy[i, j] = torch.dot(outputs[i], embeddings[j])
        return outputs_copy

    def measure_acc_alt(self, outputs, labels):
        # output: size of [batch_size, embedding size], ideally should be the average embedding of a lot of vectors
        # labels: size of [batch_size, vocab]
        temp = outputs[0, :].squeeze()
        # print(temp.size())
        print(self.gensim.most_similar(temp))
        print(self.gensim.most_similar_cosmul(temp))
        print(self.gensim.wv[temp])
        similarWords = self.gensim.most_similar_cosmul(positive=word)
        for i in range(0, 5):
            pass



    def process_label(self, labels): # labels is in shape of [batch, vocab]
        label = torch.zeros(self.out_dimention, 1)
        for i in range(0, labels.size()[0]):
            current = torch.zeros(self.out_dimention)
            for j in range(0, labels.size()[1]):
                if labels[i, j] == 1:
                    current = current + (torch.FloatTensor(self.hashtag_to_embedding_dict[self.id_to_word[str(j)]]))
            label = torch.cat((label, current.reshape((self.out_dimention, 1))), dim=1)
        label = label.permute((1, 0))
        label = label[1:, :]
        return label
    def training(self):
        tr_loss = 0
        tr_acc = 0
        for epoch in range(self.epochs):
            # print("Epoch: ",epoch," loss: ",tr_loss," acc: ",tr_acc)
            tr_loss = 0
            tr_acc = 0
            l = 0
            self.model.train()
            for i, batch in enumerate(self.data.train_loader):

                inputs, labels=batch
                inputs = inputs.type(torch.FloatTensor)
                hash_labels = labels.type(torch.FloatTensor).squeeze()
                labels = self.process_label(hash_labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.model_name=='cnn':
                    # outputs = self.compare_with_embeddings(outputs,self.data.all_hashtags);
                    pass
                loss = self.loss_fnc(outputs.squeeze(), labels.squeeze())
                loss.backward()
                self.optimizer.step()

                tr_acc += self.measure_acc(outputs, hash_labels)
                tr_loss += loss.item()
                l += 1
            self.train_acc += [tr_acc / l]
            self.train_loss += [tr_loss / l]
            print('(Training) Epoch: ', epoch, ' loss: ', tr_loss / l, ' acc: ', tr_acc / l)
            v_acc = 0
            v_loss = 0
            l = 0
            self.model.eval()
            for j, batch in enumerate(self.data.val_loader):
                inputs, labels=batch
                inputs = inputs.type(torch.FloatTensor)
                hash_labels = labels.type(torch.FloatTensor).squeeze()
                labels = self.process_label(hash_labels)
                outputs = self.model(inputs)
                if self.model_name=='cnn':
                    # outputs = self.compare_with_embeddings(outputs,self.data.all_hashtags)
                    pass
                v_acc += self.measure_acc(outputs, hash_labels)
                v_loss += self.loss_fnc((outputs.squeeze()), labels.squeeze()).item()
                l += 1
            self.valid_loss += [v_loss / l]
            self.valid_acc += [v_acc / l]
            print('(Validation) Epoch: ', epoch, ' loss: ', v_loss / l, ' acc: ', v_acc / l)
        pass
    def show_result(self):
        print("train acc: ", self.train_acc[-1], "train loss", self.train_loss[-1])
        print("validate acc: ", self.valid_acc[-1], "validate loss", self.valid_loss[-1])
        print(self.train_acc[-1], self.valid_acc[-1])
        print('Finished Training')
        pyplot.plot(np.array(self.train_loss), label="training set")
        pyplot.title("Loss vs Epochs")
        pyplot.ylabel("Loss")
        pyplot.legend(loc='upper right')
        pyplot.xlabel("Epoch")
        pyplot.show()
        pyplot.plot(np.array(self.train_acc), label="training set")
        pyplot.title("Accuracy vs Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.legend(loc='lower right')
        pyplot.xlabel("Epoch")
        pyplot.show()
        pyplot.plot(np.array(self.valid_loss), label="validation set")
        pyplot.title("Loss vs Epochs")
        pyplot.ylabel("Loss")
        pyplot.legend(loc='upper right')
        pyplot.xlabel("Epoch")
        pyplot.show()
        pyplot.plot(np.array(self.valid_acc), label="validation set")
        pyplot.title("Accuracy vs Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.legend(loc='lower right')
        pyplot.xlabel("Epoch")
        pyplot.show()
    def save_model(self):
        torch.save(self.model, 'model_'+self.model_name+'.pt')