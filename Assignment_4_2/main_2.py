import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from os.path import isfile, join
from os import listdir
from os import system
from PIL import Image
from matplotlib import pyplot as plt
from model import ConvolutionalNeuralNetwork, ConvolutionalNeuralNetwork2Layers, ConvolutionalNeuralNetwork4Layers, ConvolutionalNeuralNetwork1Layers, FlexibleCNN
from dataset import ImageDataset
from time import time
from torchsummary import summary
import numpy as np
from math import floor, fabs

# =============================================== Parameters ===============================================
seed = 3
lr = 0.1
numOfEpoch = 210
batch_size = 32
evalEvery = 10
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
def evaluateAccuracy(predictions, labels):
    # model.eval()
    accCounter = 0
    for i in range (0, predictions.size()[0]):
        difference = predictions[i,:].double() - torch.DoubleTensor(oneHotDict[labels[i].item()])
        difference = difference.squeeze()
        for i in range(0, difference.size()[0]):
            difference[i] = fabs(difference[i])
        ones = torch.ones(difference.size())
        zeros = torch.zeros(difference.size())
        difference = torch.where((difference) < 0.5, zeros, ones)
        if difference.sum() == 0:
            accCounter = accCounter + 1
    return accCounter/predictions.size()[0]
def evalulate(model, valLoader):
    accCounter = 0
    for i, batch in enumerate(valLoader):
        data, labels = batch
        data = data.type(torch.FloatTensor)
        predictions = model(data)
        for i in range(0, predictions.size()[0]):
            difference = predictions[i, :].double() - labels[i, :].double()
            for i in range (0, difference.size()[0]):
                difference[i] = fabs(difference[i])
            ones = torch.ones(difference.size())
            zeros = torch.zeros(difference.size())
            difference = torch.where((difference) < 0.5, zeros, ones)
            if difference.sum() == 0:
                accCounter = accCounter + 1
    # print(float(total_corr)/len(val_loader.dataset))
    return float(accCounter)/len(valLoader.dataset)
def evalulateWithLoss(model, valLoader, lossFunc):
    accCounter = 0
    lossCounter = 0
    for i, batch in enumerate(valLoader):
        data, labels = batch
        data = data.type(torch.FloatTensor)
        model.eval()
        predictions = model(data)
        lossCounter = lossCounter + lossFunc(input=predictions.squeeze(), target=labels.long())
        for i in range(0, predictions.size()[0]):
            difference = predictions[i, :].double() - torch.DoubleTensor(oneHotDict[labels[i].item()])
            difference = difference.squeeze()
            for i in range (0, difference.size()[0]):
                difference[i] = fabs(difference[i])
            ones = torch.ones(difference.size())
            zeros = torch.zeros(difference.size())
            difference = torch.where((difference) < 0.5, zeros, ones)
            if difference.sum() == 0:
                accCounter = accCounter + 1
    # print(float(total_corr)/len(val_loader.dataset))
    return (float(accCounter)/len(valLoader.dataset), loss/len(valLoader.dataset))
def isTrainingFile(path):
    for i in range(39, 46):
        if str(i) in path:
            return False
    return True

def isValidationFile(path):
    for i in range(39, 46):
        if str(i) in path:
            return True
    return False
def generateConfusionMatrix(model, valLoader):
    predictionList = []
    labelList = []
    for k, batch in enumerate(valLoader):
        data, labels = batch
        predictions = model(data)
        for i in range(0, predictions.size()[0]):
            maxVal = torch.max(predictions[i, :])
            for j in range (0, 10):
                if predictions[i, j].item() == maxVal:
                    predictionList.append(j)
                    break
            labelList.append(labels[i].item())
    print(confusion_matrix(labelList, predictionList, labels=[0,1,2,3,4,5,6,7,8,9]))

# =============================================== Misc Functions ===============================================

# ===============================================Getting the data readt================================================

myDireNames = []
myFileNames = []
for f in listdir("./asl_images"):
    if f != '.DS_Store':
        myDireNames.append("./asl_images/" + f + "/")
for dir in myDireNames:
    for img in listdir(dir):
        myFileNames.append(dir+img)
# also try to find mean and std by getting a list of the numpy version of the images
myNPImages =  []
myImages = []
for pic in myFileNames:
    if ".jpg" in pic:
        img = Image.open(pic)
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
# ===============================================Getting the data into python================================================
# =============================================== getting the data loaders ready================================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])
myValidationDataSet = torchvision.datasets.ImageFolder(root="./asl_images/", transform = transform, is_valid_file = isValidationFile)
myTrainingDataSet = torchvision.datasets.ImageFolder(root="./asl_images/", transform = transform, is_valid_file = isTrainingFile)
# instead of modifying the dataset object, I'm simply going to make a dictionary that can reference oneHOT values from  numbers here
oneh_encoder = OneHotEncoder(categories='auto')
letterDict = myTrainingDataSet.class_to_idx # from
inv_letterDict = {v: k for k, v in letterDict.items()}
oneHotDictKey = np.array(list(myTrainingDataSet.class_to_idx.values()))
oneHotDictVal = oneh_encoder.fit_transform(np.array(list(myTrainingDataSet.class_to_idx.values())).reshape((len(myTrainingDataSet.class_to_idx.values()), 1))).todense()
oneHotDict = {}
for i in range(0, oneHotDictKey.shape[0]):
    oneHotDict[oneHotDictKey[i]] = oneHotDictVal[i]
myTrainingDataLoader = DataLoader(myTrainingDataSet, batch_size=len(myTrainingDataSet), shuffle=True)
myValidationDataLoader = DataLoader(myValidationDataSet, batch_size=len(myValidationDataSet), shuffle=True)
counter = 0
trainingImageData = []
trainingLabelData = []
validationImageData = []
validationLabelData = []
# this for loop extract the label and images from the original dataloader, so I can one-hot-encode it.
for i, batch in enumerate(myTrainingDataLoader):
    feat, label = batch
    for i in range (0, label.squeeze().shape[0]):
        # trainingLabelData.append(np.array(oneHotDict[label[i].item()]))
        trainingLabelData.append(np.array(label[i].item()))
        trainingImageData.append(np.array(feat[i]))
    # try to encode the data:
for i, batch in enumerate(myValidationDataLoader):
    feat, label = batch
    for i in range (0, label.squeeze().shape[0]):
        # validationLabelData.append(np.array(oneHotDict[label[i].item()]))
        validationLabelData.append(np.array(label[i].item()))
        validationImageData.append(np.array(feat[i]))
trainingLabelData = np.array(trainingLabelData).squeeze()
trainingImageData = np.array(trainingImageData).squeeze()
validationLabelData = np.array(validationLabelData).squeeze()
validationImageData = np.array(validationImageData).squeeze()

print(trainingLabelData.shape)
print(trainingImageData.shape)
print(validationLabelData.shape)
print(validationImageData.shape)

trainingDataSet = ImageDataset(trainingImageData, trainingLabelData)
trainingDataLoader = DataLoader(trainingDataSet, batch_size=batch_size, shuffle=True)
validationDataSet = ImageDataset(validationImageData, validationLabelData)
validationDataLoader = DataLoader(validationDataSet, batch_size=batch_size, shuffle=True)
# ========================e======================= getting the data loaders ready================================================

# =============================================== training ================================================

torch.manual_seed(seed=seed)
model = ConvolutionalNeuralNetwork4Layers()
print(summary(model, input_size=(3, 56, 56)))
# lossFunc = torch.nn.MSELoss()
lossFunc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
lossTrend = []
accuracyTrend = []
epochTrend = []
vlossTrend = []
vaccuracyTrend = []
vepochTrend = []
startTime = time()
for epoch in range(0, numOfEpoch):
    for i, batch in enumerate(trainingDataLoader):
        feat, label = batch
        optimizer.zero_grad()
        prediction = model(feat)
        loss = lossFunc(input=prediction.squeeze(), target=label.long())
        loss.backward()
        optimizer.step()
        #record data
        if i == floor(trainingImageData.shape[0]/batch_size - 1):
            epochTrend.append(epoch)
            accuracyTrend.append(evaluateAccuracy(prediction, label))
            lossTrend.append(loss/batch_size)
    if epoch%evalEvery == 0:
        accuracy, loss = evalulateWithLoss(model, validationDataLoader, lossFunc)
        vaccuracyTrend.append(accuracy)
        vlossTrend.append(loss)
        vepochTrend.append(epoch)
        print("accuracy of epoch " + str(epoch) + " is " + str(accuracy))
endtime = time()
print(evalulateWithLoss(model, validationDataLoader, lossFunc)[0])
print(endtime-startTime)
generateConfusionMatrix(model, validationDataLoader)
# torch.save(model.state_dict(),"MyBestSmall.pt")
epochTrend = np.array(epochTrend)
accuracyTrend = np.array(accuracyTrend)
lossTrend  = np.array(lossTrend)
vaccuracyTrend = np.array(vaccuracyTrend)
vepochTrend = np.array(vepochTrend)
plt.subplot(2, 1, 1)
plt.title("Loss and Accuracy of training data vs Epochs")
plt.plot(epochTrend, lossTrend, label = "training")
plt.plot(vepochTrend, vlossTrend, 'bo', label = "validation")
plt.legend(loc = 'lower right')
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.subplot(2, 1, 2)
plt.plot(epochTrend, accuracyTrend, label = "training")
plt.plot(vepochTrend, vaccuracyTrend, 'bo', label = "validation")
plt.legend(loc = 'lower right')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
# =============================================== training ================================================