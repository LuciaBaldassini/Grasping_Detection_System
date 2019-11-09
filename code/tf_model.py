import tensorflow as tf
import matplotlib.pyplot as plt


class OurResNet:
    def __init__(self, pretrained, optimizer="Adam", loss="mse"):
        self.model = None
        self.img_shape = (480, 640, 3)
        if pretrained:
            weights = "imagenet"
        else:
            weights = None
        ResNet_model = tf.keras.applications.ResNet50(input_shape=self.img_shape,
                                                      include_top=False,
                                                      weights=weights)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        flatten_layer = tf.keras.layers.Flatten()
        prediction_layer = tf.keras.layers.Dense(5, activation='relu')

        # ResNet_model.trainable = False
        self.model = tf.keras.Sequential([ResNet_model, flatten_layer, prediction_layer])
        # model = tf.keras.Sequential([ResNet_model, global_average_layer, prediction_layer])

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=["accuracy"])

    def plot_training(self, history, with_validation):
        if with_validation:
            plt.plot(history["accuracy"])
            plt.plot(history["val_accuracy"])
            plt.title("ResNet50 Training Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Training set", "Validation set"], loc="upper left")
            plt.savefig("./output/model-training-accuracy")
            plt.close()

            plt.plot(history["loss"])
            plt.plot(history["val_loss"])
            plt.title("ResNet50 Training Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Training set", "Validation set"], loc="upper left")
            plt.savefig("./output/model-training-loss")
            plt.close()
        else:
            plt.plot(history["accuracy"])
            plt.title("ResNet50 Training Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.savefig("./output/model-training-accuracy")
            plt.close()

            plt.plot(history["loss"])
            plt.title("ResNet50 Training Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.savefig("./output/model-training-loss")
            plt.close()

    def train(self, x_train, y_train, epochs, batch_size, validation_split, output_path):
        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_split)
        self.plot_training(history.history, (validation_split != 0))

    def predict(self, image):
        prediction = self.model.predict(image)
        return prediction
