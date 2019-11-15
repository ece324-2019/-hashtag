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
class instagram_data_set:
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
            os.system(cmd)
            self.all_hashtags=[]
            self.all_images={}
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
                    resp = requests.get(url, stream=True)
                    local_file = open('../img/'+str(i)+'-'+str(j)+'.jpg', 'wb')
                    resp.raw.decode_content = True
                    shutil.copyfileobj(resp.raw, local_file,length=1024*1024)
                    local_file.close()
                    local_file_name = "../img/" + str(i) + '-' + str(j) + '.jpg'
                    self.all_images[local_file_name] = []
                    im = Image.open(local_file_name)
                    im = im.resize((200,200),Image.NEAREST)
                    im.save('../img/' + '/' + str(i) +'-' + str(j) + '.jpg')
                    for hashtag in hashtags:
                        if hashtag not in self.all_hashtags:
                            self.all_hashtags+=[hashtag]
                        self.all_images[local_file_name]+=[hashtag]
                    del im
                    del local_file
                    del local_file_name
                    del resp
                    j+=1
            json_file.close()
            hashtag_dct = {self.all_hashtags[i]:i for i in range(0, len(self.all_hashtags))}
            with open('hashtags.json','w') as hashtag_file:
                json.dump(hashtag_dct,hashtag_file);
            with open('images.json','w') as image_file:
                json.dump(self.all_images,image_file);
            hashtag_file.close()
            image_file.close()
        #----------the crawling is done and the images are sorted into hashtag folders-----------#

        with open('hashtags.json','r') as hashtag_file:
            self.all_hashtags=json.load(hashtag_file)
        with open('images.json','r') as image_file:
            self.all_images=json.load(image_file)
        print("Done loading images")
        # -----------------------------------raw data prepared-----------------------------------#
        array_image=np.array([])
        labels = np.array([])
        i=0;
        for local_file_name,hashtag_list in self.all_images.items():
            im = Image.open(local_file_name)
            im = np.array(im,dtype=float)
            im = np.reshape(im,(3,200,200))
            im -= im.mean(2).mean(1).mean(0)
            im /= np.reshape(im, (-1)).std()
            label = np.zeros((1,len(self.all_hashtags.values())))
            for hashtag in hashtag_list:
                label[0,self.all_hashtags[hashtag]]=1;
            if i==0:
                array_image=np.array([im])
                labels=np.array([label])
            else:
                array_image=np.concatenate((array_image, np.array([im])), axis=0)
                labels=np.concatenate((labels, np.array([label])), axis=0)
            i=1
        train_data, test_data, train_label, test_label = train_test_split(array_image, labels,
                                                                              test_size=0.2,
                                                                              random_state=0)
        train_data_set = Instagram_Dataset(train_data, train_label)
        print(array_image.shape)
        print("Mean:", array_image.mean(), "Std", array_image.std())
        test_data_set = Instagram_Dataset(test_data, test_label)
        self.train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(test_data_set, batch_size=len(test_data_set), shuffle=True)
class instagram_data_set:
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
            self.all_images={}
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
                    resp = requests.get(url, stream=True)
                    local_file = open('../img/'+str(i)+'-'+str(j)+'.jpg', 'wb')
                    resp.raw.decode_content = True
                    shutil.copyfileobj(resp.raw, local_file,length=1024*1024)
                    local_file.close()
                    local_file_name = "../img/" + str(i) + '-' + str(j) + '.jpg'
                    self.all_images[local_file_name] = []
                    im = Image.open(local_file_name)
                    im = im.resize((200,200),Image.NEAREST)
                    im.save('../img/' + '/' + str(i) +'-' + str(j) + '.jpg')
                    for hashtag in hashtags:
                        if hashtag not in self.all_hashtags:
                            self.all_hashtags+=[hashtag]
                        self.all_images[local_file_name]+=[hashtag]
                    del im
                    del local_file
                    del local_file_name
                    del resp
                    j+=1
            json_file.close()
            hashtag_dct = {self.all_hashtags[i]:i for i in range(0, len(self.all_hashtags))}
            with open('hashtags.json','w') as hashtag_file:
                json.dump(hashtag_dct,hashtag_file);
            with open('images.json','w') as image_file:
                json.dump(self.all_images,image_file);
            hashtag_file.close()
            image_file.close()
        #----------the crawling is done and the images are sorted into hashtag folders-----------#

        with open('hashtags.json','r') as hashtag_file:
            self.all_hashtags=json.load(hashtag_file)
        with open('images.json','r') as image_file:
            self.all_images=json.load(image_file)
        print("Done loading images")
        # -----------------------------------raw data prepared-----------------------------------#
        array_image=np.array([])
        labels = np.array([])
        i=0;
        for local_file_name,hashtag_list in self.all_images.items():
            im = Image.open(local_file_name)
            im = np.array(im,dtype=float)
            im = np.reshape(im,(3,200,200))
            im -= im.mean(2).mean(1).mean(0)
            im /= np.reshape(im, (-1)).std()
            label = np.zeros((1,len(self.all_hashtags.values())))
            for hashtag in hashtag_list:
                label[0,self.all_hashtags[hashtag]]=1;
            if i==0:
                array_image=np.array([im])
                labels=np.array([label])
            else:
                array_image=np.concatenate((array_image, np.array([im])), axis=0)
                labels=np.concatenate((labels, np.array([label])), axis=0)
            i=1
        train_data, test_data, train_label, test_label = train_test_split(array_image, labels,
                                                                              test_size=0.2,
                                                                              random_state=0)
        train_data_set = Instagram_Dataset(train_data, train_label)
        print(array_image.shape)
        print("Mean:", array_image.mean(), "Std", array_image.std())
        test_data_set = Instagram_Dataset(test_data, test_label)
        self.train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(test_data_set, batch_size=len(test_data_set), shuffle=True)

        hashtag_file.close()
        image_file.close()

object=instagram_data_set(start_user='passthekimchi',num_per_user=100,recraw=False,system='windows',batch_size=100)
#pass