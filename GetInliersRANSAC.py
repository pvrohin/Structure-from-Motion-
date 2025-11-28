import random
import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInliersRANSAC(points1, points2, index):
    """
    Rejects the outliers from a set of feature matches and returns the inliner indices.

    Parameters
    ----------
    points1 : numpy.ndarray
        feature points for matching in first image
    points2 : numpy.ndarray
        feature points for matching in second image
    index : numpy.ndarray
        index of all feature matches

    Results
    -------
    inlier_index : numpy.ndarray
        index of all inlier feature matches
    outlier_index : numpy.ndarray
        index of all outlier feature matches for visulaization
    F_best : numpy.ndarray
        the best fundamental matrix
    """
    