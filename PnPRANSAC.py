import random
import numpy as np
from LinearPnP import LinearPnP
    

def PnPRANSAC(X, x, K):
    """
    Estimates the 6-DoF camera pose with respect to a 3D object using the Perspective-n-Point (PnP) algorithm
    with Random Sample Consensus (RANSAC) to handle outliers.

    Parameters
    ----------
    X : numpy.ndarray
        a set of 3D points in the world
    x : numpy.ndarray
        the 2D projections of the 3D points in the image
    K : numpy.ndarray
        the camera intrinsic matrix

    Results
    -------
    Cnew : numpy.ndarray
        the estimated center of camera
    Rnew : numpy.ndarray
        the estimated rotaion matrix
    """
    