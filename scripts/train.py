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
from torchvision import models

class train:
    def __init__(self,data,loss_function='MSELoss',model='cnn',lr=0.001,epochs=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model=='cnn' or model=='transfer':
            self.out_dimension=40
        if model=='baseline':
            self.out_dimension=len(data.all_hashtags)
        if loss_function=="MSELoss":
            self.loss_fnc=nn.MSELoss()
        if loss_function=="KLDivLoss":
            self.loss_fnc1=nn.KLDivLoss()
            self.loss_fnc2 = nn.BCEWithLogitsLoss()
        self.model=CNN(output_dim=self.out_dimension)
        self.l=len(data.all_hashtags)
        self.epochs=epochs
        self.data=data
        self.model_name=model
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []
        self.valid_f1 = []
        self.train_f1 = []
        self.optimizer=optim.Adam(self.model.parameters(), lr=lr)
        if model=='cnn':
            hashtags_dic=ht.generate_dict_of_hashtag()
            for key in self.data.all_hashtags.keys():
                try:
                    self.data.all_hashtags[key]=hashtags_dic[key]
                except:
                    self.data.all_hashtags[key]=np.zeros(self.out_dimension)
            self.embeddings = torch.tensor(np.asarray(list(self.data.all_hashtags.values())))
            self.embeddings=self.embeddings.type(torch.float32)
        if model == 'transfer':
            hashtags_dic = ht.generate_dict_of_hashtag()
            for key in self.data.all_hashtags.keys():
                try:
                    self.data.all_hashtags[key] = hashtags_dic[key]
                except:
                    self.data.all_hashtags[key] = np.zeros(self.out_dimension)
            self.embeddings = torch.tensor(np.asarray(list(self.data.all_hashtags.values())))
            self.embeddings = self.embeddings.type(torch.float32)
            self.model = models.resnet50(True)
        self.model.to(self.device)

    def measure_acc(self,outputs, labels):
        acc=0
        for i in range(0, len(outputs)):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            beta=3
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
            # acc_temp=(tp*(1+beta*beta))/(tp*(1+beta*beta)+beta*fp+fn)
            acc_temp=(tp+tn)/(tp+tn+fn+fp)
            acc+=acc_temp
        print(tp,fp,tn,fn)
        return acc/len(outputs)
    def measure_f1(self,outputs, labels):
        acc=0
        for i in range(0, len(outputs)):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            beta=2
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
            acc_temp=(tp*(1+beta*beta))/(tp*(1+beta*beta)+beta*fp+fn)
            # cc_temp=(tp+tn)/(tp+tn+fn+fp)
            acc+=acc_temp
        # print(tp,fp,tn,fn)
        return acc/len(outputs)

    def compare_with_embeddings(self,outputs):
        outputs_copy=torch.zeros([len(outputs),self.l])
        for i in range(len(outputs)):
            outputs_copy[i]=nn.functional.cosine_similarity(outputs[i],self.embeddings.to(device=self.device),dim=-1)
        #outputs_copy=nn.functional.leaky_relu(outputs_copy*2,0.01)
        outputs_copy=torch.sigmoid(outputs_copy)
        return outputs_copy

    def training(self):
        print("Start training")
        # if torch.cuda.is_available():
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        #     print("using GPU")
        # if torch.cuda.is_available():
        #     self.model.cuda()
        #     print("using GPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        tr_loss = 0
        tr_acc = 0
        v_f1 = 0
        for epoch in range(self.epochs):
            tr_loss = 0
            tr_acc = 0
            tr_f1 = 0
            l = 0
            for i, batch in enumerate(self.data.train_loader):
                inputs, labels=batch
                inputs = inputs.type(torch.float32).to(device)
                labels = labels.type(torch.float32).to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                #print("model computation")
                if self.model_name=='cnn':
                    outputs=self.compare_with_embeddings(outputs)
                #print("comparison computation")

                loss = self.loss_fnc1(outputs.squeeze(), labels.squeeze())+self.loss_fnc2(outputs.squeeze(), labels.squeeze())*0.1
                loss.backward()
                self.optimizer.step()

                tr_acc += self.measure_acc(outputs, labels)
                tr_loss += loss.item()
                tr_f1 += self.measure_f1(outputs, labels)
                l += 1
            self.train_acc += [tr_acc / l]
            self.train_loss += [tr_loss / l]
            self.train_f1 += [tr_f1 / l]
            print('Epoch: ',epoch,' loss: ',tr_loss/l,' acc: ',tr_acc/l, ' f1: ', tr_f1/l)
            v_acc = 0
            v_loss = 0
            v_f1 = 0
            l = 0
            for j, batch in enumerate(self.data.val_loader):
                inputs, labels=batch
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                outputs = self.model(inputs)
                if self.model_name=='cnn':
                    outputs = self.compare_with_embeddings(outputs)
                v_acc += self.measure_acc(outputs, labels)
                v_f1 += self.measure_f1(outputs, labels)
                v_loss += (self.loss_fnc1(outputs.squeeze(), labels.squeeze())+0.11*self.loss_fnc2(outputs.squeeze(), labels.squeeze())).item()
                l += 1
            self.valid_loss += [v_loss / l]
            self.valid_acc += [v_acc / l]
            self.valid_f1 += [v_f1 / l]
        pass
    def show_result(self):
        print("train acc: ", self.train_acc[-1], "train loss", self.train_loss[-1])
        print("validate acc: ", self.valid_acc[-1], "validate loss", self.valid_loss[-1])
        print("train f1: ", self.train_f1[-1], "valid f1: ", self.valid_f1[-1])
        print('Finished Training')
        pyplot.plot(np.array(self.train_loss), label="training set")
        pyplot.title("Loss vs Epochs")
        pyplot.ylabel("Loss")
        pyplot.legend(loc='upper right')
        pyplot.xlabel("Epoch")
        pyplot.show()
        pyplot.plot(np.array(self.train_f1), label="training set")
        pyplot.title("F1 Score vs Epochs")
        pyplot.ylabel("F1 Score")
        pyplot.legend(loc='lower right')
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
        pyplot.plot(np.array(self.valid_f1), label="validation set")
        pyplot.title("F1 Score vs Epochs")
        pyplot.ylabel("F1 Score")
        pyplot.legend(loc='lower right')
        pyplot.xlabel("Epoch")
        pyplot.show()
    def save_model(self):
        torch.save(self.model, 'model_'+self.model_name+'.pt')
