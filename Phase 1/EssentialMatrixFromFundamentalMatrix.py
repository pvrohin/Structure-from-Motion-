import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    Computes the essential matrix from the given fundamental matrix and the camera internal matrix.

    Parameters
    ----------
    F : array-like
        The fundamental matrix
    K : array-like
        The camera internal matrix
    
    Results
    -------
    E : array-like
        The essential matrix
    """
    return K.T @ F @ K