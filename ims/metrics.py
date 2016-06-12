import cv2
import scipy.spatial.distance as sci_dist
import numpy as np


def hist_move_distance(h1, h2, power_=2):
    needed, availiable = map(list, zip(*[(y - x if x >= y else 0, y - x if y >= x else 0) for x, y in zip(h1, h2)]))
    dist = 0
    j = 0
    for i in xrange(len(needed)):
        val = needed[i]
        while val != 0:
            if availiable[j] == 0:
                j += 1
                continue
            if availiable[j] >= abs(val):
                dist += abs(val) * abs(pow(i - j, power_))
                availiable[j] += val
                val = 0
            else:
                dist += abs(availiable[j]) * abs(pow(i - j, power_))
                # print availiable
                val += availiable[j]
                availiable[j] = 0

    return -dist


def hist_angle_distance(h1, h2, **kwargs):
    def _unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = _unit_vector(h1)
    v2_u = _unit_vector(h2)
    return -np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def euclidean_distance(h1, h2, **kwargs):
    return -np.linalg.norm(h1 - h2)


def opencv_correlation_distance(h1, h2, **kwargs):
    return cv2.compareHist(np.float32(h1), np.float32(h2), cv2.cv.CV_COMP_CORREL)


def opencv_chi_squared_distance(h1, h2, **kwargs):
    return -cv2.compareHist(np.float32(h1), np.float32(h2), cv2.cv.CV_COMP_CHISQR)


def opencv_intersection_distance(h1, h2, **kwargs):
    return cv2.compareHist(np.float32(h1), np.float32(h2), cv2.cv.CV_COMP_INTERSECT)


def opencv_hellinger_distance(h1, h2, **kwargs):
    return -cv2.compareHist(np.float32(h1), np.float32(h2), cv2.cv.CV_COMP_BHATTACHARYYA)


def scipy_chebyshev(h1, h2, **kwargs):
    return -sci_dist.chebyshev(h1, h2)


def scipy_cityblock(h1, h2, **kwargs):
    return -sci_dist.cityblock(h1, h2)


def scipy_hamming(h1, h2, **jwargs):
    return -sci_dist.hamming(h1, h2)


hist_distance = hist_angle_distance
