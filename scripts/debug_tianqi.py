import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pyplot
from copy import *
import torchtext
from torchtext import data
import spacy
from gensim.models.keyedvectors import KeyedVectors
import argparse
import os
from model import *
from dataset import *
from access_word_vector import *
def measure_acc(outputs, labels):
    acc=0
    for i in range(0, len(outputs)):
        output=outputs[i].detach().numpy()
        label=labels[i].squeeze().detach().numpy()
        acc+=1-sum(np.multiply(output-label,output-label))/len(output)
    return acc/len(outputs)

def compare_with_embeddings(outputs,hashtags):
    embbeddings=returnListOfEmbedding(hashtags)
    embbeddings=torch.FloatTensor(embbeddings)
    outputs_copy=torch.zeros([len(outputs),len(hashtags)])
    for i in range(len(outputs)):
        for j in range(len(hashtags)):
            outputs_copy[i,j]=torch.dot(outputs[i],embbeddings[j])
    return outputs_copy
batch_size=100
num_epoch=50
learning_rate=0.001

data=instagram_data_set(start_user='passthekimchi',num_per_user=100,recraw=False,system='windows',batch_size=batch_size)
train_loader, test_loader=data.train_loader,data.val_loader
model, loss_fnc = CNN(output_dim=300), nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=learning_rate)
train_acc = []
train_loss = []
valid_acc = []
valid_loss = []
for epoch in range(num_epoch):
    print(epoch)
    tr_loss = 0
    tr_acc = 0
    l = 0
    for i, batch in enumerate(train_loader):
        inputs, labels=batch
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs=compare_with_embeddings(outputs,data.all_hashtags);
        loss = loss_fnc(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()

        tr_acc += measure_acc(outputs, labels)
        tr_loss += loss.item()
        l += 1
    train_acc += [tr_acc / l]
    train_loss += [tr_loss / l]
    v_acc = 0
    v_loss = 0
    l = 0
    for j, batch in enumerate(test_loader):
        inputs, labels=batch
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        outputs = model(inputs)
        outputs = compare_with_embeddings(outputs,data.all_hashtags)
        v_acc += measure_acc(outputs, labels)
        v_loss += loss_fnc((outputs.squeeze()), labels.squeeze()).item()
        l += 1
    valid_loss += [v_loss / l]
    valid_acc += [v_acc / l]
print("train acc: ", train_acc[-1], "train loss", train_loss[-1])
print("validate acc: ", valid_acc[-1], "validate loss", valid_loss[-1])
print(train_acc[-1], valid_acc[-1])
print('Finished Training')
pyplot.plot(np.array(train_loss), label="training set")
pyplot.title("Loss vs Epochs")
pyplot.ylabel("Loss")
pyplot.legend(loc='upper right')
pyplot.xlabel("Epoch")
pyplot.show()
pyplot.plot(np.array(train_acc), label="training set")
pyplot.title("Accuracy vs Epochs")
pyplot.ylabel("Accuracy")
pyplot.legend(loc='lower right')
pyplot.xlabel("Epoch")
pyplot.show()
pyplot.plot(np.array(valid_loss), label="validation set")
pyplot.title("Loss vs Epochs")
pyplot.ylabel("Loss")
pyplot.legend(loc='upper right')
pyplot.xlabel("Epoch")
pyplot.show()
pyplot.plot(np.array(valid_acc), label="validation set")
pyplot.title("Accuracy vs Epochs")
pyplot.ylabel("Accuracy")
pyplot.legend(loc='lower right')
pyplot.xlabel("Epoch")
pyplot.show()
torch.save(model, 'model_cnn.pt')