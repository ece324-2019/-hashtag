import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, kernel_num = 10, fc1_num = 10, fc2_num = 20):
        super(FlexibleCNN, self).__init__()
        self.kernel_num = kernel_num
        self.conv1 = nn.Conv2d(3, kernel_num, 3) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 10 * 10, fc1_num) # takes in 12 x 12 images
        # self.fc2 = nn.Linear(fc1_num, fc2_num)  # takes in 12 x 12 images
        self.fc3 = nn.Linear(fc1_num, 10) # output Layer
    def forward(self, input):
        x = (F.relu(self.maxpool(self.conv1(input))))
        x = (F.relu(self.maxpool(self.conv2(x))))
        x = (F.relu((self.conv3(x))))
        x = x.view(-1, self.kernel_num * 10 * 10)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x




class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(10, 10, 3) # 4 x 27 x 27 come in, 8 x 25 x 25
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(10 * 12 * 12, 8) # takes in 12 x 12 images
        self.fc2 = nn.Linear(8, 10) # output Layer
    def forward(self, input):
        x = F.relu(self.maxpool(self.conv1(input)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 10 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
class ConvolutionalNeuralNetwork1Layers(nn.Module):
    def __init__(self, kernel_num = 30, fc1_num = 32):
        super(ConvolutionalNeuralNetwork1Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, kernel_num, 3) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 27 * 27, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, 10) # output Layer
    def forward(self, input):
        x = F.relu(self.maxpool(self.conv1(input)))
        x = x.view(-1, 30 * 27 * 27)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
class ConvolutionalNeuralNetwork2Layers(nn.Module):
    def __init__(self, kernel_num = 30, fc1_num = 32):
        super(ConvolutionalNeuralNetwork2Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, kernel_num, 3) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(kernel_num, kernel_num, 3) # 4 x 27 x 27 come in, 8 x 25 x 25
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 12 * 12, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, 10) # output Layer
    def forward(self, input):
        x = F.relu(self.maxpool(self.conv1(input)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 30 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
class SmallConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, kernel_num = 7, fc1_num = 18):
        super(SmallConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, kernel_num, 3) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 5 * 5, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, 10) # output Layer
    def forward(self, input):
        x = F.relu(self.maxpool(self.conv1(input)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = F.relu(self.maxpool(self.conv3(x)))
        x = x.view(-1, 7 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
class ConvolutionalNeuralNetwork4Layers(nn.Module):
    def __init__(self, kernel_num = 10, fc1_num = 32):
        super(ConvolutionalNeuralNetwork4Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, kernel_num, 3) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.conv4 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 8 * 8, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, 10) # output Layer
    def forward(self, input):
        # x = self.conv_bn(F.relu(self.maxpool(self.conv1(input))))
        # x = self.conv_bn(F.relu(self.maxpool(self.conv2(x))))
        # x = self.conv_bn(F.relu((self.conv3(x))))
        # x = self.conv_bn(F.relu((self.conv4(x))))
        x = (F.relu(self.maxpool(self.conv1(input))))
        x = (F.relu(self.maxpool(self.conv2(x))))
        x = (F.relu((self.conv3(x))))
        x = (F.relu((self.conv4(x))))
        x = x.view(-1, 10 * 8 * 8)
        # x = self.fc1_bn(F.relu(self.fc1(x)))
        x = (F.relu(self.fc1(x)))
        x = F.sigmoid(self.fc2(x))
        return x