import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch


def to_four_points(rectangle):
    center_x, center_y, w, h, angle = rectangle
    half_w = w / 2
    half_h = h / 2
    v1 = [half_w * np.cos(angle), half_w * np.sin(angle)]
    v2 = [-half_h * np.sin(angle), half_h * np.cos(angle)]
    p0 = np.asarray((center_x, center_y))
    p1 = p0 - v1 - v2
    p2 = p0 + v1 - v2
    p3 = p0 + v1 + v2
    p4 = p0 - v1 + v2
    new_row = [np.round(p1).astype(int),
               np.round(p2).astype(int),
               np.round(p3).astype(int),
               np.round(p4).astype(int)]
    return new_row


def plot_image(image, rectangle=None):
    """Show image with rectangle"""
    plt.imshow(image)
    if rectangle is not None:
        rectangle = to_four_points(rectangle)
        p1, p2, p3, p4 = rectangle
        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='gray')
        plt.plot((p2[0], p3[0]), (p2[1], p3[1]), color='green')
        plt.plot((p3[0], p4[0]), (p3[1], p4[1]), color='gray')
        plt.plot((p4[0], p1[0]), (p4[1], p1[1]), color='green')
    plt.pause(0.001)  # pause a bit so that plots are updated


def calculate_similarity(predicted, labels, device):
    similarities = []
    for i, label in enumerate(labels):
        min_sum = torch.zeros(1, device=predicted.device)
        max_sum = torch.zeros(1, device=predicted.device)
        for x in zip(predicted[i], label):
            x = torch.Tensor(x)
            min_sum += x.min()
            max_sum += x.max()
        similarity = min_sum / max_sum
        similarities.append(similarity)
    similarities = torch.stack(similarities)
    return similarities.mean().abs()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Grasping detection system')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers to fetch data (default: 4)')

    parser.add_argument('--test_split', type=float, default=0.1,
                        help='percentage of dataset to use as test set (default: 0.1)')

    parser.add_argument('--valid_split', type=int, default=0.1,
                        help='percentage of dataset to use as test set (default: 0.1)')

    parser.add_argument('--test_and_plot', type=str, default="",
                        help='The path to the saved model we want test network and plot rectangles on images ('
                             'default: "")')
    args = parser.parse_args()
    return (args.batch_size, args.epochs, args.num_workers, args.test_split,
            args.valid_split, args.test_and_plot)
