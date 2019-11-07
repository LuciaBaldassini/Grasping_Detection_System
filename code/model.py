import os
from keras_applications import resnet
from keras.models import load_model


def load(filename):
    try:
        model = load_model(filename)
        return model
    except IOError as e:
        print("No such file")


class ResNet50:

    def __init__(self, pretrained=False):
        self.pre_trained = pretrained
        self._build()

    def _build(self):
        if self.pre_trained:
            self.model = resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=5)
        else:
            self.model = resnet.ResNet50(include_top=True, weights='None', input_tensor=None, input_shape=None, pooling=None, classes=5)
        return self.model

    def summary(self):
        print()
        print()
        print("RESNET50 MODEL")
        print("--------------------")
        self.model.summary()

    def save_model(self, path):
        print("saving ResNet50 model...")
        if not os.path.isdir(str(path)):
            os.mkdir(str(path))

        self.model.save(str(path) + "/resNet50.h5")
