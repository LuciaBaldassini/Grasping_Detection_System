from __future__ import print_function, division
import time
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from cornell_dataset import CornellDataset, ToTensor, Normalize, de_normalize
from util import plot_image, parse_arguments
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


def train_network(epochs, n_train_batches, n_val_batches, n_test_batches):
    # Training
    metrics = []
    best_val_accuracy = 0
    current_test_accuracy = 0

    appendix_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    filename = 'our_resnet18_' + appendix_datetime + '.pt'
    saved_model = PATH_TO_OUTPUTS / filename
    filename = 'metrics_' + appendix_datetime
    saved_metrics = PATH_TO_OUTPUTS / filename

    start_ts = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = model.train()
        val_losses, val_accuracies = model.validate()
        val_accuracy = sum(val_accuracies) / n_val_batches
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_model(saved_model)
            _, test_accuracy = model.test()
            current_test_accuracy = sum(test_accuracy) / n_test_batches

        print(f"Epoch {epoch}/{epochs}, time elapsed: {(time.time() - start_ts):.2f}s, training loss:"
              f" {total_loss / n_train_batches}, validation loss: {sum(val_losses) / n_val_batches} -- validation "
              f"accuracy: {val_accuracy}, test accuracy: {current_test_accuracy}")

        metrics.append((total_loss / n_train_batches, sum(val_losses) / n_val_batches, val_accuracy,
                        current_test_accuracy))  # for plotting learning curve

    print(f"Total training time: {(time.time() - start_ts):.2f}s")
    model.save_experiment(saved_metrics, metrics)

    print("Finished training")


if __name__ == '__main__':

    batch_size, epochs, num_workers, test_split, valid_split, test_and_plot = parse_arguments()
    # ROOT_PATH = Path('/home/diego/Documents/RUG/CognitiveRobotics/Grasping_Detection_System')
    # PATH_TO_DATA = ROOT_PATH / 'debug_dataset'
    ROOT_PATH = Path('/home/s3736555/Grasping_Detection_System')
    PATH_TO_DATA = ROOT_PATH / 'dataset'
    PATH_TO_POS_LABELS = ROOT_PATH / 'labels/pos_labels.csv'
    PATH_TO_OUTPUTS = ROOT_PATH / 'output'
    # Make sure output exists
    if not PATH_TO_OUTPUTS.exists():
        Path.mkdir(PATH_TO_OUTPUTS, parents=True)
    BATCH_SIZE = batch_size
    NUM_WORKERS = num_workers
    TEST_SPLIT = test_split
    VALIDATION_SPLIT = valid_split
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
    model = OurResnet(dest_path=PATH_TO_OUTPUTS,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      test_loader=test_loader,
                      pre_trained=PRE_TRAINED)
    # print(model.model)

    if not test_and_plot == "":
        path = Path(test_and_plot)
        device = torch.device('cpu')
        model.load_model(path, device=device)
        images, predictions = model.get_prediction(test_loader)
        for i, batch in enumerate(predictions):
            for j, rect in enumerate(batch):
                image = images[i][j]
                image = de_normalize(image)
                image = image.numpy().transpose((1, 2, 0))
                # print(f"Predicted rectangles {rect}")
                plot_image(image, rect)

    else:
        print(f"Starting Training, batch_size: {batch_size}, epochs: {epochs}, num_workers: {num_workers}, "
              f"test_split: {test_split}, valid_split: {valid_split}")
        n_train_batches = len(train_loader)
        n_val_batches = len(valid_loader)
        n_test_batches = len(test_loader)
        train_network(epochs, n_train_batches, n_val_batches, n_test_batches)

    print("Bye")
