import torch
from PIL import Image
from access_word_vector import *
import numpy as np
import json
model=torch.load('model_.pt')
local_file_name="../0-3-0.jpg"
im = Image.open(local_file_name)
im = im.resize((200,200),Image.NEAREST)
im = np.array(im,dtype=float)
im = np.reshape(im, (3, 200, 200))
im -= im.mean(2).mean(1).mean(0)
im /= np.reshape(im, (-1)).std()
with open('hashtags.json', 'r') as hashtag_file:
    hashtags = json.load(hashtag_file)
input=torch.unsqueeze(torch.from_numpy(im),0)
input = input.type(torch.FloatTensor)
output=model(input)
pass