import numpy as np
import matplotlib.pyplot as plt


def to_four_points(rectangle):
    center_x, center_y, w, h, angle = rectangle
    half_w = w / 2
    half_h = h / 2
    v1 = [half_w * np.cos(angle), half_w * np.sin(angle)]
    v2 = [-half_h * np.sin(angle), half_h * np.cos(angle)]
    p0 = np.asarray((center_x, center_y))
    p1 = p0 - v1 - v2
    p2 = p0 + v1 - v2
    p3 = p0 + v1 + v2
    p4 = p0 - v1 + v2
    new_row = [np.round(p1).astype(int),
               np.round(p2).astype(int),
               np.round(p3).astype(int),
               np.round(p4).astype(int)]
    return new_row


def plot_image(image, rectangle=None):
    """Show image with rectangle"""
    plt.imshow(image)
    if rectangle is not None:
        rectangle = to_four_points(rectangle)
        p1, p2, p3, p4 = rectangle
        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color='gray')
        plt.plot((p2[0], p3[0]), (p2[1], p3[1]), color='green')
        plt.plot((p3[0], p4[0]), (p3[1], p4[1]), color='gray')
        plt.plot((p4[0], p1[0]), (p4[1], p1[1]), color='green')
    plt.pause(0.001)  # pause a bit so that plots are updated
