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
batch_size=100
num_epoch=50
learning_rate=0.001
embedding_dim=100
data=instagram_data_set(start_user='therock',num_per_user=100,recraw=False,system='windows',batch_size=batch_size)
train_loader, test_loader = data.train_loader, data.val_loader
train_model=train(cnn_out_dimention=len(data.all_hashtags),data=data)
train_model.training()
train_model.show_result()