from utility import load_data
from preprocess import to_five_dimensional, split_train_test_data, replicate_images, normalize_pix_val

from model import ResNet50

path = "../debug_dataset"  # Only add a couple of pictures to this path
images, pos_rectangles, neg_rectangles = load_data(path)
pos_rectangles = to_five_dimensional(pos_rectangles)
replicated_images = replicate_images(images, pos_rectangles)
x_train, y_train, x_test, y_test = split_train_test_data(replicated_images, pos_rectangles)

x_train, x_test = normalize_pix_val(x_train), normalize_pix_val(x_test)

model = ResNet50(pretrained=True)
model.summary()

import numpy as np
x_train = x_train["images"]
x_train = np.array([x for x in x_train])
del y_train["filenames"]

model.train(x_train, y_train, epochs=20, batch_size=1, validation_split=0.1, output_path="./output")
#model.test(x_train, y_train, "./output", batch_size=1)