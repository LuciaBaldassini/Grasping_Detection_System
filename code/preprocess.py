import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

# DEBUG
from utility import load_data, plot


def to_four_points(rectangles):
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
        new_row = [filename, p1, p2, p3, p4]
        df.loc[len(df)] = new_row

    return df


def to_five_dimensional(corner_points):
    df = pd.DataFrame(columns=['filenames', 'center_x', 'center_y', 'width', 'height', 'angle'])
    for filename in corner_points.filenames.unique():
        points = corner_points[corner_points["filenames"] == filename]
        points = points.loc[:, ['x', 'y']]
        points.astype('float64')
        for i in range(0, len(points), 4):
            p1 = np.asarray((points.iloc[i][0], points.iloc[i][1]))
            p2 = np.asarray((points.iloc[i + 1][0], points.iloc[i + 1][1]))
            p3 = np.asarray((points.iloc[i + 2][0], points.iloc[i + 2][1]))
            # p4 = np.asarray((points.iloc[i + 3][0], points.iloc[i + 3][1]))
            w = euclidean(p1, p2)
            h = euclidean(p2, p3)
            v_diag = p3 - p1  # the vector representing the diagonal
            # v_diag = (v_diag / np.linalg.norm(v_diag)) * 0.5 * euclidean(p1, p3)
            center = p1 + v_diag / 2
            angle = np.arcsin((p2[1] - p1[1]) / w)
            # degrees = np.rad2deg(angle)  # DEBUG only
            new_row = [filename, center[0], center[1], w, h, angle]
            df.loc[len(df)] = new_row
    return df


def test():
    path = "../debug_dataset"  # Only add a couple of pictures to this path
    images, pos_rectangles, neg_rectangles = load_data(path)
    pos_rectangles = to_five_dimensional(pos_rectangles)
    df = to_four_points(pos_rectangles)
    for i, j in images.iterrows():
        rectangles = df[df["filenames"] == j["filenames"]]
        plot(j["images"], j["filenames"], rectangles)


def test_without_changes():
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
            new_row = [filename, np.asarray([x1, y1]), np.asarray([x2, y2]), np.asarray([x3, y3]), np.asarray([x4, y4])]
            df.loc[len(df)] = new_row
    print(df)
    for i, j in images.iterrows():
        rectangles = df[df["filenames"] == j["filenames"]]
        plot(j["images"], j["filenames"], rectangles)


if __name__ == "__main__":
    test_without_changes()
    test()
