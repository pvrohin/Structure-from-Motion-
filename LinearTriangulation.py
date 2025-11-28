import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Computes the 3D position of a set of points given its projections in two images using
    linear triangulation.

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

    Results
    -------
    X : array-like
        set of vectors representing the 3D positions of points in space 
    """
    


