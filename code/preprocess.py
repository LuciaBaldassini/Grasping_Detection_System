import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

# DEBUG
from utility import load_data, plot


def to_four_points():
    pass


def to_five_dimensional(corner_points):
    df = pd.DataFrame(columns=['filenames', 'center_x', 'center_y', 'width', 'height', 'angle'])
    rectangle_list = []
    for filename in corner_points.filenames.unique():
        points = corner_points[corner_points["filenames"] == filename]
        del points["filenames"]
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
            df = df.append({'filenames': filename, 'center_x': center_x, 'center_y': center_y, 'width': w, 'height': h,
                            'angle': angle}, ignore_index=True)
    return df


# def test():
path = "../debug_dataset"

images, pos_rectangles, neg_rectangles = load_data(path)

bla = to_five_dimensional(pos_rectangles)

for i, j in images.iterrows():
    plot(j["images"], j["filenames"])

# test()
