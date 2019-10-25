import cv2 as cv


def plot(image, image_name):
  cv.namedWindow(image_name, cv.WINDOW_NORMAL)
  cv.imshow(image_name, image)
  cv.waitKey(0)
  cv.destroyWindow(image_name)


def read_image(image_path):
  return cv.imread(image_path, cv.IMREAD_UNCHANGED)

def test():
  image_path = "../dataset/pcd0100r.png"
  image_name = "test"
  image = read_image(image_path)
  plot(image, image_name)

test()
