import os

import numpy as np
from cv2 import cv2

EXTENSIONS = ['jpg', 'jpeg', 'png']


def get_images_in_dir(d):
    """get the relative path to all of the images in the given directory

    :param d: the directory to look in relative to the running process
    :type d: str
    :return: all of the image paths
    :rtype: list[str]
    """
    # list to store the paths to the images
    paths = []

    # get all the files in the directory
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            paths.append(full_path)

    # filter out the files that don't have typeical image extensions
    image_paths = [f for f in paths if f.split('.')[-1] in EXTENSIONS]

    return image_paths


def flip_kernel(kernel):
    """flip a kernel (probably to prep for convolution).

    :param img: the kernel
    :type img: matrix
    """
    k = np.copy(kernel)

    return(cv2.flip(k, -1))


def convolve(image, kernel):
    """convolve the image with the kernel.

    :param img: the image to convolve
    :type img: cv2 image
    :param kernel: the kernel to convolve with
    :type kernel: matrix
    """
    img = np.copy(image)

    return(cv2.filter2D(img, -1, flip_kernel(kernel)))
