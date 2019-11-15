import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, kernel_num = 10, fc1_num = 100, output_dim=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, kernel_num, 2) # 3 x 56 x 56 comes in, 4 x 54 x 54 come out
        self.conv2 = nn.Conv2d(kernel_num, kernel_num, 3)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, 4)
        self.conv4 = nn.Conv2d(kernel_num, kernel_num, 5)
        self.maxpool = nn.MaxPool2d(2, 2) # after conv one, makes 4 x 27 x 27, after conv two, makes 8 x 12 x 12
        self.fc1 = nn.Linear(kernel_num * 9 * 9, fc1_num) # takes in 12 x 12 images
        self.fc2 = nn.Linear(fc1_num, output_dim) # output Layer
    def forward(self, input):
        x = (F.relu(self.maxpool(self.conv1(input))))
        x = (F.relu(self.maxpool(self.conv2(x))))
        x = (F.relu(self.maxpool(self.conv3(x))))
        x = (F.relu(self.maxpool(self.conv4(x))))
        x = x.view(-1, 10 * 9 * 9)
        x = (F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x