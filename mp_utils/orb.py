import numpy as np
from cv2 import cv2
import mp_utils.vis as vis
import mp_utils.convert as convert


def quick_orb(test, exemplar, min_match_count=4):

    # create an ORB object
    orb = cv2.ORB_create()

    # convert both images to GRAYSCALE
    exemplar_g = convert.bgr_to_gray(exemplar)
    test_g = convert.bgr_to_gray(test)

    # Find the keypoints and descriptors with ORB
    kpts1, descs1 = orb.detectAndCompute(exemplar_g, None)
    kpts2, descs2 = orb.detectAndCompute(test_g, None)

    # create a brute force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match the descriptors
    matches = bf.match(descs1, descs2)

    # sort in order of distance
    sorted_matches = sorted(matches, key=lambda x:x.distance)

    return sorted_matches