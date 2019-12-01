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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    y_true = y_true.cpu().detach().numpy().round()
    y_pred = ((y_pred.cpu().detach().numpy() > 0.5) * 1).round()
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
class train:
    def __init__(self,data,loss_function='MSELoss',model='cnn',lr=0.001,epochs=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Linear(num_ftrs, self.out_dimension)
        if model == 'baseline':
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Linear(num_ftrs, self.out_dimension)

        self.model.to(self.device)

    def measure_acc(self,outputs, labels):
        outputs = outputs.cpu()
        labels = labels.cpu()
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
        outputs = outputs.cpu()
        labels = labels.cpu()
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
        outputs_copy=torch.sigmoid(10*outputs_copy)
        # outputs_copy=torch.relu(outputs_copy)
        return outputs_copy

    def training(self):
        print("Start training")
        # if torch.cuda.is_available():
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        #     print("using GPU")
        # if torch.cuda.is_available():
        #     self.model.cuda()
        #     print("using GPU")
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
                inputs = inputs.type(torch.float32).to(self.device)
                labels = labels.type(torch.float32).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).to(self.device)
                #print("model computation")
                if self.model_name=='cnn' or self.model_name == "transfer":
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
                inputs = inputs.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)
                outputs = self.model(inputs).to(self.device)
                if self.model_name=='cnn' or self.model_name == "transfer":
                    outputs = self.compare_with_embeddings(outputs).to(self.device)
                v_acc += self.measure_acc(outputs, labels)
                v_f1 += self.measure_f1(outputs, labels)
                v_loss += (self.loss_fnc1(outputs.squeeze(), labels.squeeze())+0.11*self.loss_fnc2(outputs.squeeze(), labels.squeeze())).item()
                l += 1
            self.valid_loss += [v_loss / l]
            self.valid_acc += [v_acc / l]
            self.valid_f1 += [v_f1 / l]
        pass
    def show_result(self, folder_path = "./"):
        print("train acc: ", self.train_acc[-1], "train loss", self.train_loss[-1])
        print("validate acc: ", self.valid_acc[-1], "validate loss", self.valid_loss[-1])
        print("train f1: ", self.train_f1[-1], "valid f1: ", self.valid_f1[-1])
        print('Finished Training')
        pyplot.plot(np.array(self.train_loss), label="training set")
        pyplot.plot(np.array(self.valid_loss), label="validation set")
        pyplot.title("Loss vs Epochs")
        pyplot.ylabel("Loss")
        pyplot.legend(loc='upper right')
        pyplot.xlabel("Epoch")
        pyplot.savefig(folder_path + "both_loss")
        pyplot.clear()
        pyplot.show()
        pyplot.plot(np.array(self.train_f1), label="training set")
        pyplot.plot(np.array(self.valid_f1), label="validation set")
        pyplot.title("F1 Score vs Epochs")
        pyplot.ylabel("F1 Score")
        pyplot.legend(loc='lower right')
        pyplot.xlabel("Epoch")
        pyplot.savefig(folder_path + "both_F1")
        pyplot.show()
        pyplot.clear()
        pyplot.plot(np.array(self.train_acc), label="training set")
        pyplot.plot(np.array(self.valid_acc), label="validation set")
        pyplot.title("Accuracy vs Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.legend(loc='lower right')
        pyplot.xlabel("Epoch")
        pyplot.savefig(folder_path + "both_accuracy")
        pyplot.show()
        pyplot.clear()
        # pyplot.plot(np.array(self.valid_loss), label="validation set")
        # pyplot.title("Loss vs Epochs")
        # pyplot.ylabel("Loss")
        # pyplot.legend(loc='upper right')
        # pyplot.xlabel("Epoch")
        # pyplot.savefig(folder_path + "validation_loss")
        # pyplot.show()
        # pyplot.plot(np.array(self.valid_acc), label="validation set")
        # pyplot.title("Accuracy vs Epochs")
        # pyplot.ylabel("Accuracy")
        # pyplot.legend(loc='lower right')
        # pyplot.xlabel("Epoch")
        # pyplot.savefig(folder_path + "validation_accuracy")
        # pyplot.show()
        # pyplot.plot(np.array(self.valid_f1), label="validation set")
        # pyplot.title("F1 Score vs Epochs")
        # pyplot.ylabel("F1 Score")
        # pyplot.legend(loc='lower right')
        # pyplot.xlabel("Epoch")
        # pyplot.savefig(folder_path + "validation_f1")
        pyplot.show()
    def save_model(self, folder_path = './'):
        torch.save(self.model, folder_path+'model_'+self.model_name+'.pt')
