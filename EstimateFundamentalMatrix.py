import numpy as np

def EstimateFundamentalMatrix(points1, points2):
    """
    Estimates the fundamental matrix from the eight randomly selected feature matches.

    Parameters
    ----------
    points1 : array-like
        points for matching from image 1 
    points2 : array-like
        points for matching from image 2

    Results
    -------
    F : array-like
         the resulting fundamental matrix
    """
    