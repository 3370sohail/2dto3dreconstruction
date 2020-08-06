"""
Rigid 3D Transformation
----------------------------------------------------------------------------------------------------------------

adapted from https://github.com/nghiaho12/rigid_transform_3D

"Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D,
IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987

"""

from numpy import *


# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector


def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = mean(A, axis=1)
    centroid_B = mean(B, axis=1)

    # ensure centroids are 3x1 (necessary when A or B are
    # numpy arrays instead of numpy matrices)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - tile(centroid_A, (1, num_cols))
    Bm = B - tile(centroid_B, (1, num_cols))

    H = Am @ transpose(Bm)

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

