import numpy as np
from scipy.optimize import least_squares


def NonLinearTriangulation(K, C1, R1, C2, R2, x1, x2, x0):
    """
    Computes the 3D position of a set of points given its projections in two images using
    non-linear triangulation.

    Parameters
    ----------
    K : array-like
        camera inrinsic matrix
    C1 : array-like
        center of first camera
    R1 : array-like
        rotation matrix of first camera
    C2 : array-like
        center of second camera
    R2 : array-like
        rotation matrix of second matrix
    x1 : array-like
        projections of a set of points in first image 
    x2 : array-like
        projections of a set of points in second image
    X : array-like
        a set of 3D points
    Results
    -------
    X : array-like
        linearly triangulated points
    """
    
    

def Loss(X, x1, x2, P1, P2):
    """
    Loss function for optimization in non-linear triangulation.

    Parameters
    ----------
    X : array-like
        linearly triangulated points
    x1 : array-like
        projecttion of a point in first image
    x2 : array-like
        projection of a point in the second image
    P1 : array-like
        first projection matrix
    P2 : array-like
        second projection matrix

    Results
    -------
    error : numpy.ndarray
        the error array calculated for optimization 
    """
    