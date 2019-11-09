import numpy as np

from utility import load_data
from preprocess import to_five_dimensional, split_train_test_data, replicate_images, normalize_pix_val, \
    to_four_points_ndarray, img_df_to_ndarray

from tf_model import OurResNet
from utility import plot_images_with_rect

path = "../debug_dataset"  # Only add a couple of pictures to this path
print("Loading data from: {}".format(path))
images, pos_rectangles, neg_rectangles = load_data(path)
pos_rectangles = to_five_dimensional(pos_rectangles)
replicated_images = replicate_images(images, pos_rectangles)
x_train, y_train, x_test, y_test = split_train_test_data(replicated_images, pos_rectangles)
x_train, x_test = normalize_pix_val(x_train), normalize_pix_val(x_test)

x_train = img_df_to_ndarray(x_train)
y_train = y_train.loc[:, ['center_x', 'center_y', 'width', 'height', 'angle']].to_numpy()
x_test = img_df_to_ndarray(x_test)
y_test = y_test.loc[:, ['center_x', 'center_y', 'width', 'height', 'angle']].to_numpy()
print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

model = OurResNet(pretrained=True)
model.model.summary()
model.train(x_train, y_train, epochs=200, batch_size=32, validation_split=0.1, output_path="./output")

# Save model
model.model.save('my_model.h5')

# """To see some predictions"""
# test_img = x_test
# print(test_img.shape)
# # test_img = np.expand_dims(test_img, axis=0)  # If it is only one image
# print(test_img.shape)
# prediction = model.predict(test_img)
# rect = to_four_points_ndarray(prediction)
# plot_images_with_rect(x_test, rect)
