import numpy as np
from cv2 import cv2
from . import filters, vis, utils, convert


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
    # compute the median of the single channel pixel intensities
    median = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - epsilon) * median))
    upper = int(min(255, (1.0 + epsilon) * median))
    edged = cv2.Canny(image, lower, upper)

    return edged


def apply_canny(image, low_thresh, high_thresh):
    """apply the Canny edge detector

    :param image: input image
    :type image: cv2 image
    :param low_thresh: lower line threshold for hysteresis
    :type low_thresh: integer
    :param high_thresh: higher line threshold for hysteresis
    :type high_thresh: integer
    :return: image of the detected edges
    :rtype: cv2 image
    """
    edged = cv2.Canny(image, low_thresh, high_thresh)

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


def draw_hough_lines_gray(image, lines):
    """draw the hough lines onto the image for visualisation.

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
            cv2.line(img, (x1, y1), (x2, y2), 255, 3)

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
    params.minThreshold = 5
    params.maxThreshold = 220
    params.thresholdStep = 1

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 150

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


def apply_find_contours(image, n_points, min_area, n_candidates):
    """find the contours in the image.

    finds the contours in the image that are approximated by an n_points
    polygon, and are greater than min_area. it will sort the contours based on
    initial area but only approximate polygons for the larget n_candidates.

    :param image: input image
    :type image: cv2 image
    :param n_points: number of points the approximated polygon should have
    :type n_points: integer
    :param min_area: minimum area the polygon should have
    :type min_area: integer
    :param n_candidates: number of contours to check
    :type n_candidates: integer
    :return: contours that meet the point and area requirements
    :rtype: list[points]
    """
    # RETR_EXTERNAL returns the contours from the top level heirarchy
    _, cnts, _ = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort based on the contour area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:n_candidates + 1]

    # list to store the four pointed contours
    n_point_conts = []

    # process each contour
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.017 * peri, True)
        area = cv2.contourArea(approx)

        # check the number of points and the area
        if len(approx) == n_points and area > min_area:
            # # change it into a sensible array of n * 2 i.e. n_points * (x, y)
            # approx = approx.reshape(n_points, 2)
            # # convert to float32
            # approx = np.float32(approx)
            # append to the list of found contours
            n_point_conts.append(approx)

    return n_point_conts


def draw_contours(image, contours, color=(0, 255, 0), thickness=-1):
    """draw the contours onto the image.

    :param image: input image
    :type image: cv2 image
    :param contours: contours to draw on the image
    :type contours: numpy array of points
    :return: image with the contours drawn on
    :rtype: cv2 image
    """
    img = np.copy(image)
    for cont in contours:
        cv2.drawContours(img, [cont], -1, color, thickness)

    return img


def get_diamonds(image):
    """get the points of the diamonds from the supplied GRAYSCALE image

    :param image: image in GRAYSCALE
    :type image: cv2 image
    :return: list of diamonds as contours
    :rtype: float32
    """
    # threshold the image
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 47, 1)

    # blur the image
    blurred = filters.apply_gaussian_blur(thresh, 9)
    bil = cv2.bilateralFilter(blurred, 11, 60, 60)

    # detect the edges
    canny = apply_canny(bil, 175, 190)

    # find the contours, in this case it should only find the diamonds
    return apply_find_contours(canny, 4, 80000, 5)


def isolate_diamond(image, diamond):
    """extract the diamond shape from the original image

    :param images: input image
    :type images: cv2 image
    :param diamond: contour describing the diamon
    :type diamond: array of points
    :return: extracted image
    :rtype: cv2 image
    """
    # create a mask the same size as the images
    mask = np.zeros(image.shape, np.uint8)

    # draw the diamonds on the mask
    mask = draw_contours(mask, [diamond], (255, 255, 255))

    # extract the diamond form the images
    extracted = cv2.bitwise_and(image, mask)

    return extracted


def get_color(image, mask, color_names, color_values):
    mean = cv2.mean(convert.bgr_to_lab(image), mask)[:3]
    mean = np.asarray(mean).reshape(-1, 1, 3)

    # go through our colors and check distance, shortest distance is the color
    min_dist = (None, np.inf)

    # convert the BGR color values to LAB
    color_values = convert.bgr_to_lab(color_values)

    # loop over the color values
    for i, row in enumerate(color_values):

        # get the distnace from this mean to this LAB value
        d = utils.distance(row[0], mean[0][0])

        # if the distance is smaller, then upddate the smallest
        if d < min_dist[1]:
            min_dist = (i, d)

    # index of the color name was save in [0]
    color = color_names[min_dist[0]]
    return color
