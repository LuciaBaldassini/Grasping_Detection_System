from __future__ import print_function, division
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from cornell_dataset import CornellDataset, ToTensor

# PATH_TO_DATA = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/dataset')
PATH_TO_DATA = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/debug_dataset')
PATH_TO_POS_LABELS = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System/labels/pos_labels.csv')
BATCH_SIZE = 2
NUM_WORKERS = 4

cpos_labels = pd.read_csv(PATH_TO_POS_LABELS, index_col=0)

transformed_dataset = CornellDataset(PATH_TO_POS_LABELS.as_posix(), PATH_TO_DATA.as_posix(), transform=ToTensor())

dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS)

for i_batch, sample_batched in enumerate(dataloader):
    aux = sample_batched['image'].numpy()[0]
    print(i_batch, sample_batched['image'].size(), sample_batched['rectangle'].size())


print("Bye")

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i, sample['image'].size(), sample['rectangle'].size())
#
#     if i == 3:
#         break


# fig = plt.figure()
# for i in range(len(dataset)):
#     sample = dataset[i]
#
#     print(i, sample['image'].shape, sample['rectangle'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     # image, rectangle = sample
#     plot_image(**sample)
#     # show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#         break
