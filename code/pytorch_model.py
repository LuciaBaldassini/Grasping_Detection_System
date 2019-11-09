import torchvision


class OurResNet18:
    def __init__(self, pre_trained=True):
        self.model_resnet = torchvision.models.resnet18(pretrained=pre_trained)