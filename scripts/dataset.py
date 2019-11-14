import os
import json
import requests
import shutil
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import torchvision.datasets as dataset
import torchvision.transforms as transforms
class Instagram_Dataset(data.Dataset):
    def __init__(self, X, y):
        self.data=X
        self.label=y
        self.transform = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x,y

class instgram_data_set:
    def __init__(self,batch_size,start_user,num_per_user,num_user=0,recraw=True,system='windows'):
        if recraw:
            if system=='windows':
                cmd = 'python ..\crawler\crawler.py posts_full -u ' + start_user + ' -n ' + str(
                    num_per_user)+' -o ..\crawler\output.json --fetch_hashtags'
                file_path='../crawler/output.json'
            elif system=='linux':
                cmd = 'python ../crawler/crawler.py posts_full -u ' + start_user + ' -n ' + str(
                    num_per_user) + ' -o ../crawler/output.json --fetch_hashtags'
                file_path='../crawler/output.json'
            else:
                print("OS should only be windows or linux")
                0/0
            os.system(cmd)
        self.all_hashtags=[]
        with open('../crawler/output.json',errors='ignore', encoding='utf8') as json_file:
            posts = json.load(json_file)
        l=len(posts)
        shutil.rmtree('../img',ignore_errors=True)
        if system=='windows':
            os.system('mkdir ..\img')
        else:
            os.system('mkdir ../img')
        for i in range(l):
            try:
                posts[i]['hashtags'],posts[i]['img_urls']
            except:
                continue
            hashtags=posts[i]['hashtags']
            j=0
            for url in posts[i]['img_urls']:
                k=0
                for hashtag in hashtags:
                    if hashtag not in self.all_hashtags:
                        self.all_hashtags+=[hashtag]
                    resp = requests.get(url, stream=True)
                    try:
                        local_file = open('../img/'+hashtag+'/'+str(i)+'-'+str(j)+str(k)+'.jpg', 'wb')
                    except:
                        os.mkdir("../img/"+hashtag)
                        local_file = open('../img/' + hashtag + '/' + str(i) +'-' + str(j) + str(k) + '.jpg', 'wb')
                    resp.raw.decode_content = True
                    shutil.copyfileobj(resp.raw, local_file,length=1024*1024)
                    local_file.close()

                    local_file_name = "../img/" + hashtag + '/' + str(i) + '-' + str(j) + str(k) + '.jpg'
                    im = Image.open(local_file_name)
                    im = im.resize((200,200),Image.NEAREST)
                    im.save('../img/' + hashtag + '/' + str(i) +'-' + str(j) + str(k) + '.jpg')
                    del im
                    del local_file
                    del local_file_name
                    del resp
                    k+=1
                j+=1
            i+=1
        json_file.close()
        #----------the crawling is done and the images are sorted into hashtag folders-----------#
        print("Done downloading and transforming images")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))])
        self.train_loader, self.val_loader, self.testinput, self.testlabel = self.preprocessing(transform, batch_size, self.all_hashtags)

    def preprocessing(self,transform,batch_size,hashtags):
        dataloader = DataLoader(dataset.ImageFolder(root="../img", transform=transform), batch_size=1)
        classes = tuple(hashtags)

        input_dataset = np.array([])
        label_dataset = np.array([])
        for i, batch in enumerate(dataloader):
            input, label = batch
            input = input.numpy()
            label = np.array([label.numpy()])
            temp = np.zeros((1,len(hashtags)))
            temp[0, label[0, 0]] += 1
            label = temp
            if i == 0:
                input_dataset = input
                label_dataset = label
                continue
            input_dataset = np.concatenate((input_dataset, input), axis=0)
            label_dataset = np.concatenate((label_dataset, label), axis=0)

        train_data, test_data, train_label, test_label = train_test_split(input_dataset, label_dataset, test_size=0.2,
                                                                          random_state=0)
        train_data_set = Instagram_Dataset(train_data, train_label)
        print(input_dataset.shape)
        print("Mean:", input_dataset.mean(), "Std", input_dataset.std())
        test_data_set = Instagram_Dataset(test_data, test_label)
        train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data_set, batch_size=len(test_data_set), shuffle=True)
        return train_data_loader, test_data_loader, test_data, test_label



object=instgram_data_set(start_user='juventus',num_per_user=100,recraw=True,system='windows',batch_size=100)
pass