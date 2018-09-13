import numpy as np
from cv2 import cv2


def apply_gaussian_blur(image, n, border=0):
    """apply a gaussian blur to the image.

    :param img: the loaded image
    :type img: cv2 image
    :param n: the size of the kernel n x n
    :type n: int
    :param border: what to do when the kernel overlaps the border, defaults to 0
    :param border: int, optional
    :return: copy of the blurred image
    :rtype: cv2 image
    """
    img = np.copy(image)

    return cv2.GaussianBlur(img, (n, n), border)


def normalise(image):
    """normalise an images values.

    :param img: the loaded image in grayscale
    :type img: cv2 image
    :return: copy of the normalised image
    :rtype: cv2 image
    """
    img = np.copy(image)

    maximum = np.max(img)

    img = np.absolute(img)

    img = img * (255.0 / maximum)

    return img
