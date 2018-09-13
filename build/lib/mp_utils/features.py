import numpy as np
from cv2 import cv2


def apply_harris_corners(image):
    """run the harris corner detection function and dilate to increase point
       size.

    :param img: the loaded image
    :type img: cv2 image
    :return: copy of the image with the corner detections
    :rtype: cv2 image
    """

    # copy the image
    img = np.copy(image)

    # convert to gray before corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get the corners
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    corners = cv2.dilate(corners, None)

    # put the corners on the image
    img[corners > 0.01 * corners.max()] = [0, 0, 255]

    return img


def apply_shi_tomasi_corners(image):
    """apply the shi_tomasi corner detection algorithm.

    :param img: the loaded image
    :type img: cv2 image
    :return: copy of the image with detected corners marked
    :rtype: cv2 image
    """

    # copy the image
    img = np.copy(image)

    # convert to gray before corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # employ the shi-tomasi corner algorithm to detect corners
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.05, 20)

    # for each corner draw it on the original image!
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, [0, 0, 255], -1)

    return img


def apply_auto_canny(image, epsilon=0.33):
    """apply the Canny edge detector with automatic thresholding.

    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    :param image: the input grayscale input image
    :type image: cv2 image
    :param epsilon: threshold modifier, defaults to 0.33
    :param epsilon: float, optional
    """
    img = np.copy(image)
    # compute the median of the single channel pixel intensities
    median = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - epsilon) * median))
    upper = int(min(255, (1.0 + epsilon) * median))
    edged = cv2.Canny(img, lower, upper)

    # return the edged image
    return edged


def draw_hough_lines(image, lines):
    """draw the hough lines onto the image for visualisation

    :param lines: array of points that describe lines x1, y1, x2, y2
    :type lines: numpy array
    :param image: the image that the lines were calculated from
    :type image: cv2 image
    :return: the image with the hough lines drawn
    :rtype: cv2 image
    """
    img = np.copy(image)

    # draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return img


def apply_blob_detector(image):
    """apply the simple blob detection algorithm

    :param image: the image to do do blob detection on
    :type image: cv2 image
    :return: the blobs from the image
    :rtype: cv2 blobs
    """

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 100
    params.thresholdStep = 5

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.75

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    blobs = detector.detect(image)

    return blobs


def draw_blob_circles(image, blobs):
    """draws the blobs from the detection on to the image for visualisation

    https://www.learnopencv.com/blob-detection-using-opencv-python-c/

    :param blobs: the blobs from cv2s blob detector
    :type blobs: cv2 blobs
    :param image: the image to draw the blobs onto
    :type image: cv2 image
    :return: the image with the blobs as circles
    :rtype: cv2 image
    """
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img = cv2.drawKeypoints(image, blobs, np.array(
        []), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img
