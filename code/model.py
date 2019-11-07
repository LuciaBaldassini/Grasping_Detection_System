import os
from keras.applications import resnet
from keras import Input, Model
from keras.layers import Dense, Flatten
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})


class ResNet50:

  def __init__(self, pretrained=False, optimizer="Adam", loss="mse"):
    self.model = None
    self.input = Input(shape=(640, 480, 3))
    self.build(pretrained, optimizer, loss)

  def build(self, pretrained, optimizer=optimizer, loss=loss):
    if pretrained:
      model = resnet.ResNet50(include_top=False, input_tensor=self.input, weights="imagenet", classes=5)
      flat = Flatten()(model.outputs)
      dense = Dense(5, activation='relu')(flat)
      self.model = Model(inputs=model.inputs, outputs=dense)
    else:
      model = resnet.ResNet50(include_top=False, input_tensor=self.input, weights=None, classes=5)
      flat = Flatten()(model.outputs)
      dense = Dense(5, activation='relu')(flat)
      self.model = Model(inputs=model.inputs, outputs=dense)

    self.model.compile(loss=loss, optimizer=optimizer)

  def train(self, x_train, y_train, epochs, batch_size, validation_split, output_path):
    training = self.model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    if not os.path.isdir(str(output_path)):
      os.mkdir(str(output_path))

    plt.plot(training.history["acc"])
    plt.title("Pretrained ResNet50 training accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig("./output/model-training-accuracy")
    plt.close()

    plt.plot(training.history["loss"])
    plt.title("Pretrained ResNet50 training loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("./output/model-training-loss")
    plt.close()

  def test(self, x_test, y_test, output_path):
    testing = self.model.evaluate(x_test, y_test)

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
    print("saving ResNet50 model...")
    if not os.path.isdir(str(path)):
      os.mkdir(str(path))

    self.model.save(str(path) + "/resNet50.h5")
