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
class train:
    def __init__(self,data,loss_function='MSELoss',model='cnn',lr=0.001,epochs=100):
        if model=='cnn':
            self.out_dimension=40
        if model=='baseline':
            self.out_dimension=len(data.all_hashtags)
        if loss_function=="MSELoss":
            self.loss_fnc=nn.MSELoss()
        if loss_function=="CrossEntropy":
            self.loss_fnc=nn.functional.binary_cross_entropy_with_logits
        self.model=CNN(output_dim=self.out_dimension)
        self.l=len(data.all_hashtags)
        self.epochs=epochs
        self.data=data
        self.model_name=model
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []
        self.optimizer=optim.Adam(self.model.parameters(), lr=lr)
        hashtags_dic=ht.generate_dict_of_hashtag()
        for key in self.data.all_hashtags.keys():
            try:
                self.data.all_hashtags[key]=hashtags_dic[key]
            except:
                self.data.all_hashtags[key]=np.zeros(self.out_dimension)
        self.embeddings = torch.tensor(np.asarray(list(self.data.all_hashtags.values())))
        self.embeddings=self.embeddings.type(torch.float32)
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
            #acc_temp=(1+beta*beta)*tp/((1+beta*beta)*tp+beta*beta*fn+fp)
            acc_temp=(tp+tn)/(tp+tn+fn+fp)
            acc+=acc_temp
        return acc/len(outputs)

    def compare_with_embeddings(self,outputs):
        outputs_copy=torch.zeros([len(outputs),self.l])
        for i in range(len(outputs)):
            for j in range(self.l):
                temp=torch.dot(outputs[i],self.embeddings[j])/torch.norm(outputs[i])/torch.norm(self.embeddings[j])
                if (not np.isnan(temp.detach())):
                    outputs_copy[i,j]=temp
                else:
                    outputs_copy[i, j] = 0
        return outputs_copy

    def training(self):
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        if torch.cuda.is_available():
            self.model.cuda()
        tr_loss = 0
        tr_acc = 0
        for epoch in range(self.epochs):
            tr_loss = 0
            tr_acc = 0
            l = 0
            for i, batch in enumerate(self.data.train_loader):
                inputs, labels=batch
                inputs = inputs.type(torch.float32)
                labels = labels.type(torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.model_name=='cnn':
                    outputs=self.compare_with_embeddings(outputs)
                loss = self.loss_fnc(outputs.squeeze(), labels.squeeze())
                loss.backward()
                self.optimizer.step()

                tr_acc += self.measure_acc(outputs, labels)
                tr_loss += loss.item()
                l += 1
            self.train_acc += [tr_acc / l]
            self.train_loss += [tr_loss / l]
            v_acc = 0
            v_loss = 0
            l = 0
            for j, batch in enumerate(self.data.val_loader):
                inputs, labels=batch
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                outputs = self.model(inputs)
                print("model computation")
                if self.model_name=='cnn':
                    outputs = self.compare_with_embeddings(outputs)
                print("comparison computation")
                v_acc += self.measure_acc(outputs, labels)
                v_loss += self.loss_fnc((outputs.squeeze()), labels.squeeze()).item()
                l += 1
            self.valid_loss += [v_loss / l]
            self.valid_acc += [v_acc / l]
            print('Epoch: ',epoch,' loss: ',tr_loss,' acc: ',tr_acc)
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