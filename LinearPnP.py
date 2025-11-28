import numpy as np

def LinearPnP(Xset, xset, K):
    """
    Estimates camera pose using linear least squares method on the 3D points and corresponding 2D projection on
    the image.

    Parameters
    ----------
    Xset : numpy.ndarray
        set of 3D points
    xset : numpy.ndarray
        set of 2D projections of 3D points on the image
    K : numpy.ndarray
        camera intrinsic matrix
    
    Results
    -------
    C : numpy.ndarray
        the estimated center of camera
    R : numpy.ndarray
        the estimated rotation matrix
    """

    