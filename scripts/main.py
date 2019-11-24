#------------------------------------This is where you should start------------------------#
# Parameters:
#   start_user: Which user to start crawing
#   num_per_user: How many posts to craw per user
#   recraw: Need craw new data?
#   system: 'windows'/'linux'
#   batch_size: default=100
#   model_name: 'baseline'/'cnn'
#   embedding dimension: If using 'cnn', specify the embedding dimension of hashtag vectors
#   learning_rate: default=0.001
#   loss_function: default='MSELoss'
#   epochs: default=100

from model import *
from dataset import *
from access_word_vector import *
from train import train
import hashtag_trainer as ht

batch_size=100
num_epoch=50
learning_rate=0.001
embedding_dim=300
with open('FoodGramers.txt', 'r') as file:
    u_list = file.readlines()
data = instagram_data_set(batch_size=64,username_list=['therock\n'],num_per_user=100,recraw=False)
ht.training()
train_loader, test_loader = data.train_loader, data.val_loader
train_model=train(cnn_out_dimention=len(data.all_hashtags),data=data,epochs=30,loss_function='CrossEntropy',model='cnn')
train_model.training()
train_model.show_result()
train_model.save_model()