import inspect
import time
from pathlib import Path

import torchvision
from torch import nn, optim
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
    def __init__(self, dest_path, train_loader, valid_loader, test_loader, pre_trained=True, **kwargs):
        self.dest_path = dest_path
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.model = ResNet18(pre_trained=pre_trained)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        # See if we use CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()

    def train(self):
        total_loss = 0
        self.model.train()
        if self.cuda_available:
            self.model.cuda()
        for i, data in enumerate(self.train_loader):
            X, y = data[0].to(self.device), data[1].to(self.device)
            # training step for single batch
            self.model.zero_grad()
            outputs = self.model(X)

            loss = self.loss_function(outputs, y)
            loss.backward()
            self.optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss

    def test(self, data_loader):
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(X)  # this get's the prediction from the network

                val_losses += self.loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction

                # TODO: probably here use our own distance function
                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(self.calculate_metric(metric, y.cpu(), predicted_classes.cpu()))
        return val_losses, precision, recall, f1, accuracy

    def validate(self):
        return self.test(self.valid_loader)

    @staticmethod
    def calculate_metric(metric_fn, true_y, pred_y):
        # multi class problems need to have averaging method
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)

    def free(self):
        del self.model
        del self.train_loader
        del self.valid_loader
        del self.test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
