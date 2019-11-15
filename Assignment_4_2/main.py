import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from os.path import isfile, join
from os import listdir
from os import system
from PIL import Image
from matplotlib import pyplot as plt
from model import ConvolutionalNeuralNetwork
from dataset import ImageDataset
from time import time
from torchsummary import summary
import numpy as np
from math import fabs
# =============================================== Parameters ===============================================
seed = 2
lr = 0.01
numOfEpoch = 900
batch_size = 10
# =============================================== Parameters ===============================================
# =============================================== Misc Functions ===============================================
# load the images in ./Mypics
def organizeFolders(myImages, myFileNames, path):
    # make folders and organize into different folders depends on letters
    current = myFileNames[0].split("_")[1]
    system("mkdir " + path + current)
    for i in range(0, len(myImages)):
        splited = myFileNames[i].split("_")
        if splited[1] != current:
            current = splited[1]
            system("mkdir " + path + current)
        myImages[i].save(path + current + "/" + splited[1] + splited[2])
# show four images in four subplots
def imshowFour(imgList, labelTitle, mean = 2, std = 0.5): # should have 4 images
    coord = [(2,2,1), (2,2,2), (2,2,3), (2,2,4)]
    counter = 0
    for img in imgList:
        npimg = img.numpy()
        plt.subplot(coord[counter][0],coord[counter][1],coord[counter][2])
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(labelTitle[counter])
        counter = counter + 1
    plt.show()
# will calculate accuracy given prediction and labels
def evaluate(predictions, labels):
    accCounter = 0
    for i in range (0, predictions.size()[0]):
        difference = predictions[i,:].double() - labels[i,:].double()
        for i in range(0, difference.size()[0]):
            difference[i] = fabs(difference[i])
        ones = torch.ones(difference.size())
        zeros = torch.zeros(difference.size())
        difference = torch.where((difference) < 0.5, zeros, ones)
        if difference.sum() == 0:
            accCounter = accCounter + 1
    return accCounter/predictions.size()[0]
# =============================================== Misc Functions ===============================================

# ===============================================Getting the data readt================================================

myFileNames = []
for f in listdir("./MyPics/"):
    if isfile(join("./MyPics/", f)):
        myFileNames.append("./MyPics/" + f)
myImages = []
# also try to find mean and std by getting a list of the numpy version of the images
myNPImages =  []
for pic in myFileNames:
    if ".jpg" in pic:
        img = Image.open(pic)
        myImages.append(img)
        npIMG = np.array(img)
        myNPImages.append(npIMG)
# calculate mean and std of the images to normalize, these are actually not necessary
myNPImages = np.array(myNPImages)
means = [0,0,0]
stds = [0,0,0]
for i in range (0, 3):
    means[i] = np.mean(myNPImages[:,:,:,i])/255
    stds[i] = np.std(myNPImages[:,:,:,i])/255
# load images into folders, I've already called it
# organizeFolders(myImages, myFileNames, "./MyProcessedPics/")
myTensorImages = []
# ===============================================Getting the data into python================================================
# =============================================== getting the data loaders ready================================================

# getting training Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])
myDataSet = torchvision.datasets.ImageFolder(root="./MyProcessedPics/", transform = transform)
# instead of modifying the dataset object, I'm simply going to make a dictionary that can reference oneHOT values from  numbers here
oneh_encoder = OneHotEncoder(categories='auto')
oneHotDictKey = np.array(list(myDataSet.class_to_idx.values()))
oneHotDictVal = oneh_encoder.fit_transform(np.array(list(myDataSet.class_to_idx.values())).reshape((len(myDataSet.class_to_idx.values()), 1))).todense()
oneHotDict = {}
for i in range(0, oneHotDictKey.shape[0]):
    oneHotDict[oneHotDictKey[i]] = oneHotDictVal[i]
myDataLoader = DataLoader(myDataSet, batch_size=30, shuffle=True)
counter = 0
imageData = []
labelData = []
# this for loop extract the label and images from the original dataloader, so I can one-hot-encode it.
for i, batch in enumerate(myDataLoader):
    feat, label = batch
    for i in range (0, label.squeeze().shape[0]):
        labelData.append(np.array(oneHotDict[label[i].item()]))
        imageData.append(np.array(feat[i]))
    # try to encode the data:
labelData = np.array(labelData).squeeze()
imageData = np.array(imageData).squeeze()
myDataSet = ImageDataset(imageData, labelData)
myDataLoader = DataLoader(myDataSet, batch_size=batch_size, shuffle=True)

# =============================================== getting the data loaders ready================================================

# =============================================== training ================================================
torch.manual_seed(seed=seed)
model = ConvolutionalNeuralNetwork()
lossFunc = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
print(summary(model, input_size=(3, 56, 56)))
lossTrend = []
accuracyTrend = []
epochTrend = []
startTime = time()
for epoch in range(0, numOfEpoch):
    for i, batch in enumerate(myDataLoader):
        feat, label = batch
        optimizer.zero_grad()
        prediction = model(feat)
        loss = lossFunc(input=prediction.squeeze(), target=label.float())
        loss.backward()
        optimizer.step()
        #record data
        if i == labelData.shape[0]/batch_size - 1:
            epochTrend.append(epoch)
            accuracyTrend.append(evaluate(prediction, label))
            lossTrend.append(loss)
endtime = time()
print(endtime-startTime)
epochTrend = np.array(epochTrend)
accuracyTrend = np.array(accuracyTrend)
lossTrend  = np.array(lossTrend)

plt.subplot(2, 1, 1)
plt.title("Loss and Accuracy of training data vs Epochs")
plt.plot(epochTrend, lossTrend)
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.subplot(2, 1, 2)
plt.plot(epochTrend, accuracyTrend)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
# =============================================== training ================================================