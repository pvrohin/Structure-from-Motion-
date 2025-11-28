import numpy as np

def ExtractCameraPose(E):
    """
    Estimate the camera pose given the essentil matrix.

    Parameters
    ----------
    E : array-like
        The essential matrix
    
    Results
    -------
    Cset : array-like
        the set of camera centers
    Rset : array-like
        the set of rotation matrices
    """
    