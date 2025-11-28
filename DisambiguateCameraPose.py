import numpy as np

def DisambiguateCameraPose(Cset, Rset, Xset):
    """
    Find unique camera pose by checking the cheirality condition.

    Parameters
    ----------
    Cset : array-like
        the set of camera centers
    Rset : array-like
        the set of rotation matrices
    Xset : array-like
        set of vectors representing the 3D positions of points in space 
    
    Results
    -------
    best_C : array-like
        the corrected camera center
    best_R : array-like
        the corrected rotation matrix
    best_X : array-like
        the corrected set of vector representing the 3D positions of points in space 
    """
    