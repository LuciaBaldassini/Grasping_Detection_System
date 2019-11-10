from __future__ import print_function, division
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from cornell_dataset import CornellDataset, ToTensor, Normalize, de_normalize
from util import plot_image
from pytorch_model import OurResnet


def test_data_loader(loader):
    for i_batch, sample_batched in enumerate(loader):
        for i, image in enumerate(sample_batched['image']):
            rect = sample_batched['rectangle'][i].numpy()
            image = de_normalize(image)
            image = image.numpy().transpose((1, 2, 0))
            print("For i_batch {}, image_idx {}: {} {}".format(i_batch, i, image.shape, rect.shape))
            plot_image(image, rect)

        print(i_batch, sample_batched['image'].size(), sample_batched['rectangle'].size())
        if i_batch >= 1:
            break


# PATH_TO_DATA = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/dataset')
ROOT_PATH = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System')
PATH_TO_DATA = ROOT_PATH / 'debug_dataset'
PATH_TO_POS_LABELS = ROOT_PATH / 'labels/pos_labels.csv'
PATH_TO_OUTPUTS = ROOT_PATH / 'output'
BATCH_SIZE = 2
NUM_WORKERS = 4
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
PRE_TRAINED = True

cpos_labels = pd.read_csv(PATH_TO_POS_LABELS, index_col=0)

transformed_dataset = CornellDataset(PATH_TO_POS_LABELS.as_posix(), PATH_TO_DATA.as_posix(),
                                     transform=transforms.Compose([
                                         ToTensor(),
                                         Normalize(PRE_TRAINED)
                                     ]))

# Creating data indices for training, test and validation splits:
dataset_size = len(transformed_dataset)
indices = list(range(dataset_size))
split = int(np.floor(TEST_SPLIT * dataset_size))
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
split = int(np.floor(VALIDATION_SPLIT * len(train_indices)))
train_indices, valid_indices = train_indices[split:], train_indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                          sampler=train_sampler, num_workers=NUM_WORKERS)
valid_loader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                          sampler=valid_sampler, num_workers=NUM_WORKERS)
test_loader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                         sampler=test_sampler, num_workers=NUM_WORKERS)

# test_data_loader(train_loader)

# Create model
epochs = 100
model = OurResnet(dest_path=PATH_TO_OUTPUTS,
                  epochs=epochs,
                  train_loader=train_loader,
                  valid_loader=valid_loader,
                  test_loader=test_loader,
                  pre_trained=PRE_TRAINED)
print("The final model is")
print(model.model)

# Training
EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
    model.train()
    model.validate()

saved_model = PATH_TO_OUTPUTS / 'our_resnet18.pt'
model.save_model(saved_model)

print("Bye")
