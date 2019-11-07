#from utility import load_data
#from preprocess import to_five_dimensional, split_train_test_data, replicate_images
from model import ResNet50
"""
path = "../debug_dataset"  # Only add a couple of pictures to this path
images, pos_rectangles, neg_rectangles = load_data(path)
pos_rectangles = to_five_dimensional(pos_rectangles)
replicated_images =replicate_images(images, pos_rectangles)

x_train, y_train = split_train_test_data(replicated_images)
"""
model = ResNet50(pretrained=True)
model.summary()

model.train(x_train, y_train, epochs=50, batch_size=8, validation_split=10, output_path="./output")
model.save_model("./output")
