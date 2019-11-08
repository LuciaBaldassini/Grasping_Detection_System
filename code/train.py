import numpy as np

from utility import load_data
from preprocess import to_five_dimensional, split_train_test_data, replicate_images, normalize_pix_val

from model import ResNet50

path = "../debug_dataset"  # Only add a couple of pictures to this path
images, pos_rectangles, neg_rectangles = load_data(path)
pos_rectangles = to_five_dimensional(pos_rectangles)
replicated_images = replicate_images(images, pos_rectangles)
x_train, y_train, x_test, y_test = split_train_test_data(replicated_images, pos_rectangles)

x_train, x_test = normalize_pix_val(x_train), normalize_pix_val(x_test)

aux = x_train['images'].to_numpy()
aux = [img for img in aux]
x_train = np.asarray(aux)
# print("Shape of x_train: {}".format(x_train.shape))
y_train = y_train.loc[:, ['center_x', 'center_y', 'width', 'height', 'angle']].to_numpy()
# print("Shape of y_train: {}".format(y_train.shape))

model = ResNet50(pretrained=True)
model.summary()

model.train(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1, output_path="./output")
# model.test(x_train, y_train, "./output", batch_size=1)
