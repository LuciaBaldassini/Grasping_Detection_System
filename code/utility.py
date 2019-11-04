import cv2 as cv
import pandas as pd
import os


def plot(image, image_name):
  cv.namedWindow(image_name, cv.WINDOW_NORMAL)
  cv.imshow(image_name, image)
  cv.waitKey(0)
  cv.destroyWindow(image_name)


def read_image(image_path):
  return cv.imread(image_path, cv.IMREAD_UNCHANGED)


def load_data(data_path):
  print("loading data")

  # Determine the suffixes of the files.
  image_suffix = ".png"
  pos_suffix = "pos.txt"
  neg_suffix = "neg.txt"

  # Load the images and rectangles.
  images = pd.DataFrame(columns=["filenames", "images"])
  pos_rectangles = pd.DataFrame(columns=["filenames", "x", "y"])
  neg_rectangles = pd.DataFrame(columns=["filenames", "x", "y"])
  for filename in os.listdir(data_path + "/"):
      file_path = data_path + "/" + str(filename)
      if not os.stat(file_path).st_size == 0:

        if str(filename).endswith(image_suffix):
          new_row = [str(filename), read_image(file_path)]
          images.loc[len(images)] = new_row

        if str(filename).endswith(pos_suffix) or str(filename).endswith(neg_suffix):
          points = pd.read_csv(file_path, sep=" ", header=None)

          # Remove any extra columns with NaN values that should not be read.
          if len(points.columns) > 2:
            for i in range(2, len(points.columns)):
              del points[i]

          points.columns = ["x", "y"]
          points.insert(0, "filenames", str(filename))

          if str(filename).endswith(pos_suffix):
            pos_rectangles = pos_rectangles.append(points, ignore_index=True)
          else:
            neg_rectangles = neg_rectangles.append(points, ignore_index=True)

  return images, pos_rectangles, neg_rectangles


def test():
  path = "../dataset"

  x, y, z = load_data(path)
  print(y)
  print(z)
  print(x)
  for i, j in x.iterrows():
    plot(j["images"], j["filenames"])


test()
