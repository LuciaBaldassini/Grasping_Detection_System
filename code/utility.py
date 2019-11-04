import cv2 as cv
import pandas as pd
import os

COLORS = {"green": (0, 255, 0), "gray": (120, 120, 120), "red": (0, 0, 255), "blue": (255, 0, 0)}


def plot(image, image_name, rectangles=None):
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    if rectangles is not None:
        for _, row_values in rectangles.iterrows():
            _, p1, p2, p3, p4 = row_values
            cv.line(image, tuple(p1.tolist()), tuple(p2.tolist()), COLORS["gray"])
            cv.line(image, tuple(p2.tolist()), tuple(p3.tolist()), COLORS["green"])
            cv.line(image, tuple(p3.tolist()), tuple(p4.tolist()), COLORS["gray"])
            cv.line(image, tuple(p4.tolist()), tuple(p1.tolist()), COLORS["green"])
    cv.imshow(image_name, image)
    cv.waitKey(0)
    cv.destroyWindow(image_name)


def read_image(image_path):
    return cv.imread(image_path, cv.IMREAD_UNCHANGED)


def load_data(data_path):
    print("loading data")

    # Determine the suffixes of the files.
    image_suffix = "r.png"
    pos_suffix = "cpos.txt"
    neg_suffix = "cneg.txt"

    # Load the images and rectangles.
    images = pd.DataFrame(columns=["filenames", "images"])
    pos_rectangles = pd.DataFrame(columns=["filenames", "x", "y"])
    neg_rectangles = pd.DataFrame(columns=["filenames", "x", "y"])
    for filename in os.listdir(data_path + "/"):
        file_path = data_path + "/" + str(filename)
        if not os.stat(file_path).st_size == 0:
        
            if str(filename).endswith(image_suffix):
                suffix = str(filename).replace(image_suffix, '')
                new_row = [suffix, read_image(file_path)]
                images.loc[len(images)] = new_row

            if str(filename).endswith(pos_suffix) or str(filename).endswith(neg_suffix):
                points = pd.read_csv(file_path, sep=" ", header=None)

                # Remove any extra columns with NaN values that should not be read.
                if len(points.columns) > 2:
                    for i in range(2, len(points.columns)):
                        del points[i]

                points.columns = ["x", "y"]
                suffix = pos_suffix if str(filename).endswith(pos_suffix) else neg_suffix
                suffix = str(filename).replace(suffix, '')
                points.insert(0, "filenames", suffix)
                if str(filename).endswith(pos_suffix):
                    pos_rectangles = pos_rectangles.append(points, ignore_index=True)
                else:
                     neg_rectangles = neg_rectangles.append(points, ignore_index=True)

    return images, pos_rectangles, neg_rectangles


def test():
    path = "../test"

    x, y, z = load_data(path)
    print(y)
    print(z)
    print(x)
    for i, j in x.iterrows():
        plot(j["images"], j["filenames"])

# test()
