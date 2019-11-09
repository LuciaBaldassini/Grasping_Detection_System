import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from skimage import io


class CornellDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with the pos rectangles.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cpos_rectangles = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.image_suffix = 'r.png'
        self.cpos_suffix = 'cpos.txt'

    def __len__(self):
        return len(self.cpos_rectangles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = Path(self.root_dir) / Path(self.cpos_rectangles.iloc[idx, 0] + self.image_suffix)

        image = io.imread(img_name)
        rectangles = self.cpos_rectangles.iloc[idx, 1:]
        rectangles = rectangles.to_numpy(dtype=float)
        sample = {'image': image, 'rectangle': rectangles}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rectangle = sample['image'], sample['rectangle']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'rectangle': torch.from_numpy(rectangle)}


class Normalize(object):
    """Normalizes values between -1 and 1."""

    def __call__(self, sample):
        image, rectangle = sample['image'], sample['rectangle']
        image = image / 255.0
        return {'image': image, 'rectangle': rectangle}
