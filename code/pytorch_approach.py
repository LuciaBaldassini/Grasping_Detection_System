from __future__ import print_function, division
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from cornell_dataset import CornellDataset, ToTensor, Normalize
from util import plot_image

# PATH_TO_DATA = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/dataset')
PATH_TO_DATA = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/debug_dataset')
PATH_TO_POS_LABELS = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/labels/pos_labels.csv')
BATCH_SIZE = 2
NUM_WORKERS = 4

cpos_labels = pd.read_csv(PATH_TO_POS_LABELS, index_col=0)

transformed_dataset = CornellDataset(PATH_TO_POS_LABELS.as_posix(), PATH_TO_DATA.as_posix(),
                                     transform=transforms.Compose([
                                         ToTensor(),
                                         Normalize()
                                     ]))

dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS)

for i_batch, sample_batched in enumerate(dataloader):
    for i, image in enumerate(sample_batched['image']):
        rect = sample_batched['rectangle'][i].numpy()
        image = image.numpy().transpose((1, 2, 0))
        print("For i_batch {}, image_idx {}: {} {}".format(i_batch, i, image.shape, rect.shape))
        plot_image(image, rect)

    print(i_batch, sample_batched['image'].size(), sample_batched['rectangle'].size())
    if i_batch >= 1:
        break

print("Bye")
