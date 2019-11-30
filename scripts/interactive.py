import torch
from PIL import Image
from access_word_vector import *
import numpy as np
import torch.nn as nn
import json
import hashtag_trainer as ht


def compare_with_embeddings(outputs,l,embeddings):
    outputs_copy = torch.zeros([len(outputs), l])
    for i in range(len(outputs)):
        outputs_copy[i] = nn.functional.cosine_similarity(outputs[i], embeddings, dim=-1)
    # outputs_copy=nn.functional.leaky_relu(outputs_copy*2,0.01)
    outputs_copy = torch.sigmoid(outputs_copy)
    return outputs_copy
model=torch.load('model_cnn.pt')
hashtags_dic=ht.generate_dict_of_hashtag()
local_file_name="../6-0-0.jpg"
im = Image.open(local_file_name)
im = im.resize((200,200),Image.NEAREST)
im = np.array(im,dtype=float)
im = np.reshape(im, (3, 200, 200))
im -= im.mean(2).mean(1).mean(0)
im /= np.reshape(im, (-1)).std()
with open('hashtags.json', 'r') as hashtag_file:
    hashtags = json.load(hashtag_file)
for key in hashtags.keys():
    try:
        hashtags[key]=hashtags_dic[key]
    except:
        hashtags[key]=np.zeros(40)
embeddings = torch.tensor(np.asarray(list(hashtags.values())))
embeddings=embeddings.type(torch.float32)
input=torch.unsqueeze(torch.from_numpy(im),0)
input = input.type(torch.FloatTensor)
output=model(input)
output=compare_with_embeddings(output,len(hashtags),embeddings)
i=0
for x in output[0]:
    if x>0.5:
        print(list(hashtags.keys())[i])
    i+=1
pass