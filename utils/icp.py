"""
ICP algorithm
----------------------------------------------------------------------------------------------------------------

Most of this code from https://github.com/ClayFlannigan/icp/blob/master/icp.py

Modified by us to
    1. Be able to take in and perform ICP on point clouds of different sizes
    2. Not match 100% of points (ie. partial match)
    3. Make operations in the code to run faster. Since the code from the GitHub was mostly vectorized
       already, the main difference here is we know how many dimensions we are dealing with at all times since
       we are doing 3D ICP and not n-dimensional ICP, so we can avoid a lot of array slicing in the original code.

A quick diff check from Python's difflib's SequenceMatcher shows that our version of this ICP implementation
has roughly a 62% overlap with the original implementation on GitHub, so although there are many parts where we
are just borrowing code, we have still made some significant changes.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: 3xm numpy array of corresponding points
      B: 3xm numpy array of corresponding points
    Returns:
      T: 4x4 homogeneous transformation matrix that maps A on to B
    """
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2, :] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: 3xm array of points
        dst: 3xm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001, partial_set_size=1.):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B

    Input:
        A: 3xm numpy array of source mD points
        B: 3xm numpy array of destination mD point
        init_post: 4x4 homogenous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        partial_set_size: float in range (0, 1] which describes what portion of the point clouds have to
                          match in ICP

    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """
    # make points homogeneous, copy them to maintain the originals
    src = np.concatenate((A.T, np.ones((1, A.shape[0]))), axis=0)
    dst = np.concatenate((B.T, np.ones((1, B.shape[0]))), axis=0)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    distances = None
    iters = 1

    while iters <= max_iterations:
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:3, :].T, dst[:3, :].T)

        # partial match: only take top fraction of correspondences for matching after finding nearest neighbour
        cutoff = int(src.shape[1] * partial_set_size)
        smallest_idx = np.argpartition(distances, cutoff - 1)[:cutoff]

        distances = distances[smallest_idx]
        indices = indices[smallest_idx]
        src_smallest = src[:3, smallest_idx]

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src_smallest[:3, :].T, dst[:3, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

        iters += 1

    # calculate final transformation
    T = best_fit_transform(A, src[:3, :].T)

    return T, distances, iters
