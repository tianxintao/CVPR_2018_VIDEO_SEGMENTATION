from utils import ConvertLabelImage
import os
from skimage.io import imread,imshow,show


class VideoSegmentationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_dir, label_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_dir = train_dir
        self.root_dir = root_dir
        self.images = np.array(sorted(os.listdir(train_dir)))
        self.labels = np.array(sorted(os.listdir(label_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.images[index])
        image = imread(img_name)
        img_label = img_name = os.path.join(self.root_dir,
                                self.labels[index])
        label = imread(img_label)
        convertedLabel = ConvertLabelImage(label)
        sample = {'image': image, 'label': convertedLabel}

        return sample