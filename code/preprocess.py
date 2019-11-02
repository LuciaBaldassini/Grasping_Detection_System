from pathlib import Path

import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

# DEBUG
from utility import load_data, plot


def normalize_pix_val(image_df):
    """
    Normalize pixel values to be between 0 and 1.

    Args:
        image_df(pd.DataFrame): DataFrame containing the images. The image
            array itself should be stored in a column named "images"

    Returns:
        (pd.DataFrame): The DataFrame with the images normalized

    """
    image_df.images = image_df.images / 255.0
    return image_df


def angle_with_horizontal(v1):
    """
    Returns the angle in radians between vectors 'v1' and the horizontal

    Args:
        v1 (np.ndarray): Vector for which we want to calculate the angle.

    Returns:
        (float): The angle in radians between vector v1 and the horizontal
    """
    return np.arctan2(v1[1], v1[0])


def to_four_points(rectangles):
    """
    Transforms a 5 dimensional representation to the 4 vertices of a rectangle. Useful for ploting the rectangle on
    top of an image.

    Args:
        rectangles(pd.DataFrame): Dataframe with all the rectangles that we want to convert. Each row is a rectangle.

    Returns:
        (pd.DataFrame): Dataframe with the rectangles converted. Each row contains the four vertices.
    """
    df = pd.DataFrame(columns=["filenames", "p1", "p2", "p3", "p4"])
    for _, row_values in rectangles.iterrows():
        filename, center_x, center_y, w, h, angle = row_values
        half_w = w / 2
        half_h = h / 2
        v1 = [half_w * np.cos(angle), half_w * np.sin(angle)]
        v2 = [-half_h * np.sin(angle), half_h * np.cos(angle)]
        p0 = np.asarray((center_x, center_y))
        p1 = p0 - v1 - v2
        p2 = p0 + v1 - v2
        p3 = p0 + v1 + v2
        p4 = p0 - v1 + v2
        new_row = [filename,
                   np.round(p1).astype(int),
                   np.round(p2).astype(int),
                   np.round(p3).astype(int),
                   np.round(p4).astype(int)]
        df.loc[len(df)] = new_row

    return df


def to_five_dimensional(corner_points):
    """
    Transforms the input data read from the load_data function and converts the data points to rectangles in a 5
    dimensional representation.

    Args:
        corner_points(pd.DataFrame): The dataframe that is output from the load_data function.

    Returns:
        (pd.DataFrame): A DataFrame containing all the rectangles in a 5 dimensional representation.
            Each row is arectangle.
    """
    df = pd.DataFrame(columns=['filenames', 'center_x', 'center_y', 'width', 'height', 'angle'])
    for filename in corner_points.filenames.unique():
        points = corner_points[corner_points["filenames"] == filename]
        points = points.loc[:, ['x', 'y']]
        points.astype('float64')
        for i in range(0, len(points), 4):
            p1 = np.asarray((points.iloc[i][0], points.iloc[i][1]))
            p2 = np.asarray((points.iloc[i + 1][0], points.iloc[i + 1][1]))
            p3 = np.asarray((points.iloc[i + 2][0], points.iloc[i + 2][1]))
            p4 = np.asarray((points.iloc[i + 3][0], points.iloc[i + 3][1]))
            is_nan = np.isnan(np.asarray([p1, p2, p3, p4]))
            if not np.any(is_nan):
                w = euclidean(p1, p2)
                h = euclidean(p2, p3)
                v_diag = p3 - p1  # the vector representing the diagonal
                center = p1 + v_diag / 2
                angle = angle_with_horizontal((p2 - p1))
                new_row = [filename, center[0], center[1], w, h, angle]
                df.loc[len(df)] = new_row
    return df


def test():
    """
    Plots the rectangles on top of the image. To see if the 5-D transformation works well.
    Returns:

    """
    path = "../debug_dataset"  # Only add a couple of pictures to this path
    images, pos_rectangles, neg_rectangles = load_data(path)
    pos_rectangles = to_five_dimensional(pos_rectangles)
    df = to_four_points(pos_rectangles)
    for i, j in images.iterrows():
        rectangles = df[df.filenames == j["filenames"]]
        plot(j["images"], j["filenames"], rectangles)


def test_without_changes():
    """
    Plots the rectangles on top of the image, without converting to a 5-D representation. Useful only for debugging
    of the 5-D transform
    Returns:

    """
    path = "../debug_dataset"  # Only add a couple of pictures to this path
    images, pos_rectangles, neg_rectangles = load_data(path)
    df = pd.DataFrame(columns=["filenames", "p1", "p2", "p3", "p4"])
    for filename in pos_rectangles.filenames.unique():
        points = pos_rectangles[pos_rectangles["filenames"] == filename]
        points = points.loc[:, ['x', 'y']]
        for i in range(0, len(points), 4):
            x1, y1 = points.iloc[i][0], points.iloc[i][1]
            x2, y2 = points.iloc[i + 1][0], points.iloc[i + 1][1]
            x3, y3 = points.iloc[i + 2][0], points.iloc[i + 2][1]
            x4, y4 = points.iloc[i + 3][0], points.iloc[i + 3][1]
            new_row = [filename, np.asarray([int(x1), int(y1)]),
                       np.asarray([int(x2), int(y2)]),
                       np.asarray([int(x3), int(y3)]),
                       np.asarray([int(x4), int(y4)])]
            df.loc[len(df)] = new_row
    print(df)
    for i, j in images.iterrows():
        rectangles = df[df["filenames"] == j["filenames"]]
        plot(j["images"], j["filenames"], rectangles)


def save_labels(path_to_data, path_to_labels):
    _, pos_rectangles, neg_rectangles = load_data(path_to_data)
    pos_rectangles = to_five_dimensional(pos_rectangles)
    neg_rectangles = to_five_dimensional(neg_rectangles)

    saved_path = Path(path_to_labels)
    pos_label_path = saved_path / "pos_labels.csv"
    neg_label_path = saved_path / "neg_labels.csv"
    if not saved_path.exists():
        Path.mkdir(saved_path, parents=True)

    pos_rectangles.to_csv(pos_label_path)
    neg_rectangles.to_csv(neg_label_path)


def split_train_test_data(images_df):
    """
    Splits the images into training and test set.

    Args:
        images_df(pd.DataFrame): DataFrame with all the images in the dataset

    Returns:
        (tuple): DataFrame for the train set and test set.
    """
    df_copy = images_df.copy()
    train_set = df_copy.sample(frac=0.9, random_state=0)
    test_set = df_copy.drop(train_set.index)
    return train_set, test_set


if __name__ == "__main__":
    # test_without_changes()
    # test()
    save_labels("../dataset", "../labels")
