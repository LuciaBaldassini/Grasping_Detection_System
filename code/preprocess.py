import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

# DEBUG
from utility import load_data, plot


def to_four_points(rectangles):
    # TODO: make it so it will take a dataframe of rectangles, and output another dataframe of rectangles (size
    #  8+filename).
    df = pd.DataFrame(columns=["filenames", "p1", "p2", "p3", "p4"])
    for _, row_values in rectangles.iterrows():
        filename, center_x, center_y, w, h, angle = row_values
        v = [(w / 2) * np.cos(angle), (w / 2) * np.sin(angle)]
        v_orth = [-v[1], v[0]]
        v1 = (v / np.linalg.norm(v)) * (w / 2)
        v2 = (v_orth / np.linalg.norm(v_orth)) * (h / 2)
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
    rectangle_list = []
    for filename in corner_points.filenames.unique():
        points = corner_points[corner_points["filenames"] == filename]
        # del points["filenames"]
        points = points.loc[:, ['x', 'y']]
        # points = points.to_numpy(dtype=float)
        points.astype('float64')
        for i in range(0, len(points), 4):
            x1, y1 = points.iloc[i][0], points.iloc[i][1]
            x2, y2 = points.iloc[i + 1][0], points.iloc[i + 1][1]
            x3, y3 = points.iloc[i + 2][0], points.iloc[i + 2][1]
            # x4, y4 = points.iloc[i + 3][0], points.iloc[i + 3][1]
            w = euclidean((x1, y1), (x2, y2))
            h = euclidean((x2, y2), (x3, y3))
            center_x = x1 + w / 2
            center_y = y1 + h / 2
            angle = np.arcsin((y2 - y1) / w)
            rectangle_list.append([center_x, center_y, w, h, angle])
            new_row = [filename, center_x, center_y, w, h, angle]
            df.loc[len(df)] = new_row
    return df


def test():
    path = "../debug_dataset"  # Only add a couple of pictures to this path
    images, pos_rectangles, neg_rectangles = load_data(path)
    debug_pos_rectangles = pos_rectangles
    pos_rectangles = to_five_dimensional(pos_rectangles)
    aux = to_four_points(pos_rectangles)

    for i, j in images.iterrows():
        plot(j["images"], j["filenames"], aux)


if __name__ == "__main__":
    test()
