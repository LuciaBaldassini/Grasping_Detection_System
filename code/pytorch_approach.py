from __future__ import print_function, division
import time
import datetime
from pathlib import Path
import numpy as np
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
# Make sure output exists
if not PATH_TO_OUTPUTS.exists():
    Path.mkdir(PATH_TO_OUTPUTS, parents=True)
BATCH_SIZE = 2
NUM_WORKERS = 2
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
# print(model.model)

# Training
EPOCHS = 100
n_train_batches = len(train_loader)
n_val_batches = len(valid_loader)
n_test_batches = len(test_loader)
metrics = []
best_val_accuracy = 0
current_test_accuracy = 0

appendix_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
filename = 'our_resnet18_' + appendix_datetime + '.pt'
saved_model = PATH_TO_OUTPUTS / filename
filename = 'metrics_' + appendix_datetime
saved_metrics = PATH_TO_OUTPUTS / filename

start_ts = time.time()
# for epoch in range(1, EPOCHS + 1):
for epoch in range(1, 2 + 1):
    total_loss = model.train()
    val_losses, val_accuracies = model.validate()
    val_accuracy = sum(val_accuracies) / n_val_batches
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model.save_model(saved_model)
        _, test_accuracy = model.test()
        current_test_accuracy = sum(test_accuracy) / n_test_batches

    print(f"Epoch {epoch}/{EPOCHS}, time elapsed: {(time.time() - start_ts):.2f}s, training loss:"
          f" {total_loss / n_train_batches}, validation loss: {sum(val_losses) / n_val_batches} -- validation "
          f"accuracy: {val_accuracy}, test accuracy: {current_test_accuracy}")

    metrics.append((total_loss / n_train_batches, sum(val_losses) / n_val_batches, val_accuracy,
                    current_test_accuracy))  # for plotting learning curve

print(f"Total training time: {(time.time() - start_ts):.2f}s")
model.save_experiment(saved_metrics, metrics)

print("Bye")
