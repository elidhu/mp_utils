from . import vis, convert
import numpy as np
from cv2 import cv2


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
    sorted_matches = sorted(matches, key = lambda x:x.distance)

    # # extract the matched keypoints
    # src_pts = np.float32([kpts1[match.queryIdx].pt for match in sorted_matches]).reshape(-1,1,2)
    # dst_pts = np.float32([kpts2[match.trainIdx].pt for match in sorted_matches]).reshape(-1,1,2)

    # ## find homography matrix and do perspective transform
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # h, w = exemplar.shape[:2]
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts, M)

    # ## draw found regions
    # img2 = cv2.polylines(test, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
    # vis.show(img2, title='FOUND')

    # # draw match lines
    # res = cv2.drawMatches(exemplar, kpts1, test, kpts2, sorted_matches[:50],None,flags=2)

    # vis.show(res, title='ORB MATCH')

    # cv2.waitKey();cv2.destroyAllWindows()

    return sorted_matches