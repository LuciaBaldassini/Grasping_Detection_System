import torchvision.models as models
from torchsummary import summary
import torch
import os


def load_model():
    try:
        model = torch.load('alexNet.pt')
        return model
    except IOError as e:
        print("No such file")


class AlexNet:

    def __init__(self, pretrained=False):
        self.pre_trained = pretrained
        self._build()

    def _build(self):
        if self.pre_trained:
            self.model = models.alexnet(progress=True)
        else:
            self.model = models.alexnet(pretrained=True, progress=True)
        return self.model

    def summary(self):
        print()
        print()
        print("ALEXNET MODEL")
        print("--------------------")
        summary(self.model, (3, 224, 224, 4))

    def save_model(self, path):
        print("saving AlexNet model...")
        if not os.path.isdir(str(path)):
            os.mkdir(str(path))
        torch.save(self.model, str(path) + "alexNet.pt")
