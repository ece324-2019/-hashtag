import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, features, label):
        self.features = features
        self.label = label
    def __len__(self):
        return self.label.shape[0]
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.label[index]
        return feature, label