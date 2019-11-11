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
        rectangles = rectangles.to_numpy(dtype='float32')
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

    def __init__(self, pre_trained):
        self.pre_trained = pre_trained

    def __call__(self, sample):
        image, rectangle = sample['image'], sample['rectangle']
        image = image / 255.0
        """
        All pre-trained models expect input images normalized in the same way, 
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where 
        H and W are expected to be at least 224. The images have to be loaded in 
        to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] 
        and std = [0.229, 0.224, 0.225]
        
        Based on https://pytorch.org/docs/stable/torchvision/models.html
        """
        if self.pre_trained:
            image = image.clone()
            dtype = image.dtype
            mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=image.device)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=image.device)
            image.sub_(mean[:, None, None]).div_(std[:, None, None])

        return {'image': image, 'rectangle': rectangle}


def de_normalize(image, pre_trained=True):
    image = image.clone()
    if pre_trained:
        dtype = image.dtype
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=image.device)
        image.mul_(std[:, None, None]).add_(mean[:, None, None])
    image = image * 255.0
    return image.int()
