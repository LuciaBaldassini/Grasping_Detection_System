import time
from pathlib import Path

import torchvision
from torch import nn, optim
import torch


class ResNet18(nn.Module):
    def __init__(self, pre_trained=True):
        super().__init__()
        self.model_resnet = torchvision.models.resnet18(pretrained=pre_trained)
        print(self.model_resnet)
        self.model_resnet.fc = nn.Linear(512, 256)  # 512 for resnet18
        print(self.model_resnet)
        # self.fc1 = nn.Linear(256, 128)
        self.fc_reg = nn.Linear(128, 5)
        self.add_module('resnet', self.model_resnet)
        self.add_module('fc_reg', self.fc_reg)

    def forward(self, x):
        x = self.model_resnet(x)
        # x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc_reg(x))
        return x


class OurResnet:
    def __init__(self, dest_path, epochs, train_loader, valid_loader, test_loader, pre_trained=True, **kwargs):
        self.dest_path = dest_path
        self.epochs = epochs
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.model = ResNet18(pre_trained=pre_trained)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        # See if we use CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()

    def train(self):
        start_ts = time.time()

        metrics = []
        batches = len(self.train_loader)
        val_batches = len(self.valid_loader)
        test_batches = len(self.test_loader)
        print("batches: {}, val_batches: {}, test_batches: {}".format(batches, val_batches, test_batches))
        best_val_accuracy = 0
        current_test_accuracy = 0

        for epoch in range(self.epochs):
            total_loss = self.train()
            val_losses, precision, recall, f1, accuracy = self.validate(self.valid_loader)

            if sum(accuracy) / val_batches > best_val_accuracy:
                best_val_accuracy = sum(accuracy) / val_batches
                if self.dest_path:
                    self.save_model(self.dest_path)
                _, _, _, _, test_accuracy = self.validate(self.test_loader)
                current_test_accuracy = sum(test_accuracy) / test_batches

            print(
                f"Epoch {epoch + 1}/{self.epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
            self.print_scores(precision, recall, f1, accuracy, val_batches)
            print(f"\t{'current test accuracy'.rjust(14, ' ')}: {current_test_accuracy:.4f}")

            metrics.append((total_loss / batches, val_losses / val_batches, sum(precision) / val_batches,
                            sum(recall) / val_batches, sum(f1) / val_batches, sum(accuracy) / val_batches,
                            current_test_accuracy))  # for plotting learning curve

        print(f"Training time: {time.time() - start_ts}s")
        return metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)
