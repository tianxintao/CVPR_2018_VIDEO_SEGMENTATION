from utils import ConvertLabelImage,Resize
import torch
from torch.utils import data
import os
from skimage.io import imread,imshow,show
import numpy as np
import matplotlib.pyplot as plt

class VideoSegmentationDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_dir, label_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.images = np.array(sorted(os.listdir(train_dir)))
        self.labels = np.array(sorted(os.listdir(label_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = os.path.join(self.train_dir,
                                self.images[index])
        image = imread(img_name)
        img_label = os.path.join(self.label_dir,
                                self.labels[index])
        label = imread(img_label)
        resizedImg,resizedLabel = Resize(image,label)
        sample = {'image': torch.tensor(resizedImg,dtype=torch.float), 'label': torch.tensor(resizedLabel,dtype=torch.long)}

        return sample
    
    
if __name__ == '__main__':
    sampledTrainFolderPath = '../train_color_sample/'
    sampledLabelFolderPath = '../train_label_sample/'
    dataset = VideoSegmentationDataset(sampledTrainFolderPath,
                                    sampledLabelFolderPath)

    
    for i in range(len(dataset)):
        sample = dataset[i]
    
        print(i, sample['image'].shape, sample['label'].shape)