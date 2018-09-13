import numpy as np
from cv2 import cv2


def show_cv_image(images, title='image'):
    """show a opencv2 window that can be destroyed with any key.

    this will only work if the images are all of the same size.

    :param img: images to show
    :type img: list[cv2 image]
    :param title: title of the window to create, defaults to 'image'
    :type title: string
    """
    # if images is not a list make it one
    if type(images) is not list:
        images = [images]

    # check if the list is greater than 1 and concat the images for opencv
    if len(images) > 1:
        cv2.imshow(title, np.hstack(images))
    else:
        cv2.imshow(title, images[0])
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def gray_to_rgb(image):
    """convert cv2 image from gray to rgb for sane plotting in mpl.

    :param image: the image to be converted
    :type image: cv2 image
    :return: converted image
    :rtype: cv2 image
    """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def bgr_to_rgb(image):
    """convert form cv2s silly BGR order for sane plotting in mpl.

    :param img: the list of images to be converted
    :type img: cv2 image
    :return: list of the converted images
    :rtype: list[cv2 image]
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_gray(image):
    """convert form cv2s silly BGR order for sane plotting in mpl.

    :param img: the list of images to be converted
    :type img: cv2 image
    :return: list of the converted images
    :rtype: list[cv2 image]
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_gray32(img_path):
    """read in the image from the path as a grayscale and convert to float32.

    :param img_path: path to the image
    :type img_path: string
    :return: the loaded image
    :rtype: cv2 image
    """
    # load the image and convert to float32
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)

    return img


def read_gray(img_path):
    """read in the image from the path as a grayscale image.

    :param img_path: path to the image
    :type img_path: string
    :return: the loaded image
    :rtype: cv2 image
    """
    # load the image and convert to float32
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return img


def read_color(img_path):
    """read in the image from the path as BGR color.

    :param img_path: path to the image
    :type img_path: string
    :return: the loaded image
    :rtype: cv2 image
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    return img
