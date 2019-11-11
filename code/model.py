import os
from keras.applications import resnet
from keras import Input, Model
from keras.layers import Dense, Flatten
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})


class ResNet50:

  def __init__(self, pretrained, optimizer="Adam", loss="mse"):
    self.model = None
    self.input = Input(shape=(480, 640, 3))
    self.build(pretrained, optimizer, loss)

  def build(self, pretrained, optimizer, loss):
    if pretrained:
      weights = "imagenet"
    else:
      weights = None
    model = resnet.ResNet50(include_top=False, input_tensor=self.input, weights=weights, classes=5)
    flat = Flatten()(model.outputs)
    dense = Dense(5, activation="relu")(flat)
    self.model = Model(inputs=model.inputs, outputs=dense)

    self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

  def train(self, x_train, y_train, epochs, batch_size, validation_split, output_path):
    training = self.model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    if not os.path.isdir(str(output_path)):
      os.mkdir(str(output_path))

    if validation_split != 0:
      plt.plot(training.history["acc"])
      plt.plot(training.history["val_acc"])
      plt.title("ResNet50 Training Accuracy")
      plt.ylabel("Accuracy")
      plt.xlabel("Epoch")
      plt.legend(["Training set", "Validation set"], loc="upper left")
      plt.savefig("./output/model-training-accuracy")
      plt.close()

      plt.plot(training.history["loss"])
      plt.plot(training.history["val_loss"])
      plt.title("ResNet50 Training Loss")
      plt.ylabel("Loss")
      plt.xlabel("Epoch")
      plt.legend(["Training set", "Validation set"], loc="upper left")
      plt.savefig("./output/model-training-loss")
      plt.close()
    else:
      plt.plot(training.history["acc"])
      plt.title("ResNet50 Training Accuracy")
      plt.ylabel("Accuracy")
      plt.xlabel("Epoch")
      plt.savefig("./output/model-training-accuracy")
      plt.close()

      plt.plot(training.history["loss"])
      plt.title("ResNet50 Training Loss")
      plt.ylabel("Loss")
      plt.xlabel("Epoch")
      plt.savefig("./output/model-training-loss")
      plt.close()

  def test(self, x_test, y_test, output_path, batch_size):
    testing = self.model.evaluate(x_test, y_test, batch_size=batch_size)

    if not os.path.isdir(str(output_path)):
      os.mkdir(str(output_path))

    with open(str(output_path) + "/model-testing.txt", "w+") as output_file:
      output_file.write("Loss on test data: " + str(testing[0]))
      output_file.write("\nAccuracy on test data: " + str(testing[1]))

  def summary(self):
    print()
    print()
    print("RESNET50 MODEL")
    print("--------------------")
    self.model.summary()

  def save_model(self, path):
    print("Saving ResNet50 model")
    if not os.path.isdir(str(path)):
      os.mkdir(str(path))

    self.model.save(str(path) + "/resNet50.h5")
