import csv
import time

import cv2
import numpy as np
import open3d as o3d
from scipy import optimize
from skimage import io
from sklearn.neighbors import NearestNeighbors


def generate_keypoints_and_match(img1, img2):
    """
    Given 2 images, generate SIFT keypoints and match

    :param img1:
    :param img2:
    :return: img1 keypoints, img2 keypoints, matches
    """
    # SIFT isn't present in the newest OpenCV due to a patent, so to use it install an older version:
    #       pip install -U opencv-contrib-python==3.4.0.12
    #
    # From https://stackoverflow.com/questions/18561910/cant-use-surf-sift-in-opencv#comment97755276_47565531

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, mask=None)
    kp2, des2 = sift.detectAndCompute(img2, mask=None)

    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = matcher.match(des1, des2)

    return kp1, kp2, matches


def refine_matches(matches):
    """

    :param matches: list of matches
    :return: refined list of matches
    """
    matches = sorted(matches, key=lambda x: x.distance)

    # good_matches = []
    # for match in matches:
    matches = matches[:7]

    return matches


def pts_to_arr(pts, shape):
    """
    Write points to csv file

    :param path: path to csv file to save to
    :param shape: shape of image to write to
    :return: None
    """
    pts = pts.astype(np.int16)
    pts_arr = np.zeros(shape)

    for pt in pts:
        pts_arr[pt[1], pt[0]] = pt[2]

    return pts_arr


def get_key_points_from_matches(kp1, kp2, matches):
    """

    :param kp1: keypoints from first image
    :param kp2: keypoints from second image
    :param matches: matches
    :return: rounded coordinates of points
    """
    p1 = np.array([kp1[match.queryIdx].pt for match in matches])
    p2 = np.array([kp2[match.trainIdx].pt for match in matches])
    p1 = np.rint(p1).astype(np.int16)
    p2 = np.rint(p2).astype(np.int16)
    return p1, p2


def depth_to_voxel(img, scale=1.):
    """
    Given a depth image, convert all the points in the image to 3D points

    NOTE ON SCALE:
        The values in 3D space are not necessarily to scale. For example a car might be a meter away in
        real life, but on the depth map it only has a value of 10. We therefore need to give it a scale
        value to multiply this depth by to get its actual depth in 3D space. This scale value can be
        estimated by looking at how long or wide the actual object should be, and then scaling accordingly.

    :param img: ndarray representing depth values in image
    :param scale: how far away every value is--a number to multiply the depth values by
    :return: n x 3 ndarray, where n is the number of 3D points, and each of the 3 represents the value
             in that dimension
    """
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to n x 3
    pixels = np.stack((xx, yy, img.astype(np.int16) * scale), axis=2)
    pixels = np.reshape(pixels, (img.shape[0] * img.shape[1], 3))
    pixels = pixels[pixels[:, 2] != 0]  # filter out missing data

    return pixels


def voxel_to_csv(points, path):
    """
    Write points to csv file

    :param points: n x 3 ndarray
    :param path: path to csv file to save to
    :return: None
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(points)


def ls_array_from_xi_eq(x, y, z):
    return np.array([x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0])


def ls_array_from_yi_eq(x, y, z):
    return np.array([0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0])


def ls_array_from_zi_eq(x, y, z):
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1])


def make_homography_ls_matrix(keypoints):
    """
    use the matches to make the homography liner system matrix
    :param keypoints:
    :return: homography linear system matrix
    """
    a_matrix = []

    for i in range(len(keypoints)):
        a_matrix.append(ls_array_from_xi_eq(keypoints[i][0], keypoints[i][1], keypoints[i][2]))
        a_matrix.append(ls_array_from_yi_eq(keypoints[i][0], keypoints[i][1], keypoints[i][2]))
        a_matrix.append(ls_array_from_zi_eq(keypoints[i][0], keypoints[i][1], keypoints[i][2]))

    a_matrix = np.array(a_matrix, dtype=np.float32)
    return a_matrix


def solve_homography_ls_matrix(a_matrix):
    """
    Solve the linear system to find the homography matrix
    :param a_matrix: linear system to solve
    :return: the homography matrix
    """
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.transpose(a_matrix), a_matrix))
    h_hat = eigen_vectors[:, np.argmin(eigen_values)]

    # convert the h_hat vector into a matrix before returning it
    return h_hat.reshape(4, 4)


def homo_rigid_transform_3d(img1_kp, img2_kp):
    A = make_homography_ls_matrix(img1_kp)
    b = img2_kp.flatten()
    return constrained_least_squares(A, b)


def constrained_least_squares(A, b):
    """
    Solve for x such that ||Ax - b||_2 is minimized

    :param A: design matrix
    :param b: target vector
    :return: solution
    """
    x = optimize.lsq_linear(A, b)['x']
    x = x.reshape((3, 4))
    x = np.concatenate((x, np.array([[0, 0, 0, 1]])), axis=0)
    return x


def get_transformed_points(points, h):
    """Apply transformation to points in homogenous coordinates"""
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    transformed = np.dot(points, h.T)
    return transformed[:, :3]


def make_pcd(point_cloud):
    """

    :param point_cloud:
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd


#
#   From https://github.com/ClayFlannigan/icp/blob/master/icp.py
#


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    # get number of dimensions
    m = A.shape[1]

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
       Vt[m-1, :] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    nntime = 0

    for i in range(max_iterations):

        nnstart = time.time()
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        nntime += time.time() - nnstart

        # only take top
        cutoff = int(src.shape[1] * 0.7)
        smallest_idx = np.argpartition(distances, cutoff)[:cutoff]

        distances = distances[smallest_idx]
        indices = indices[smallest_idx]
        src_smallest = src[:m, smallest_idx]

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src_smallest[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    print("time spent calculating nearest neighbours: " + str(nntime))

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


#
#   Back to my own code
#


def apply_points_transformation(pts, transformation):
    """
    Apply a 3D homography transformation to a set of points

    :param pts: n x 3 ndarray of points
    :param transformation: 4 x 4 transformation array
    :return: transformed n x 3 points
    """
    # add ones to convert to homogenous coordinates
    ones = np.ones((pts.shape[0], 1))
    pts = np.concatenate((pts, ones), axis=1)

    pts = np.dot(transformation, pts.T).T
    return pts[:, :3] / np.expand_dims(pts[:, 3], axis=1)


if __name__ == "__main__":
    # load images
    # img1 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car1.jpg"
    # img2 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car2.jpg"

    # # load point clouds
    # img1_pts = np.genfromtxt(
    #     r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_0.csv", delimiter=",")
    # img2_pts = np.genfromtxt(
    #     r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_1.csv", delimiter=",")

    # img1_depth = pts_to_arr(img1_pts, (img1.shape[0] // 2, img1.shape[1] // 2))
    # img2_depth = pts_to_arr(img2_pts, (img2.shape[0] // 2, img2.shape[1] // 2))
    # img1_depth = img1_depth.repeat(2, axis=0).repeat(2, axis=1)
    # img2_depth = img2_depth.repeat(2, axis=0).repeat(2, axis=1)

    img1 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\kitchen_small_1_70.png"
    img2 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\kitchen_small_1_72.png"
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    img1_depth = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\kitchen_small_1_70_depth.png"
    img2_depth = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\kitchen_small_1_72_depth.png"
    img1_depth = io.imread(img1_depth)
    img2_depth = io.imread(img2_depth)

    img1_pts = depth_to_voxel(img1_depth, scale=.7)
    img2_pts = depth_to_voxel(img2_depth, scale=.7)
    voxel_to_csv(img1_pts, r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\img70.csv")
    voxel_to_csv(img2_pts, r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\img72.csv")

    # find matches
    kp1, kp2, matches = generate_keypoints_and_match(img1, img2)
    matches = refine_matches(matches)

    # img = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg=None,
    #                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("matches", img)
    # cv2.waitKey()

    # get 3D coordinates of matches
    kp1, kp2 = get_key_points_from_matches(kp1, kp2, matches)
    img1_kp = [(p[0], p[1], img1_depth[p[1], p[0]]) for p in kp1]
    img2_kp = [(p[0], p[1], img2_depth[p[1], p[0]]) for p in kp2]
    img1_kp = np.array(img1_kp)
    img2_kp = np.array(img2_kp)

    # find transformation
    h = homo_rigid_transform_3d(img1_kp, img2_kp)

    # apply transformation
    img2_new = get_transformed_points(img1_pts, h)

    np.savetxt(r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\img70_before_icp.csv",
               img2_new, delimiter=",")

    start_time = time.time()
    pts_idx1 = np.random.choice(img2_new.shape[0], img2_new.shape[0] // 30, replace=False)
    pts_idx2 = np.random.choice(img2_pts.shape[0], img2_pts.shape[0] // 30, replace=False)

    t, _, iter_to_converge = icp(img2_new[pts_idx1], img2_pts[pts_idx2], tolerance=0.001, max_iterations=100)
    print("iterations to converge: " + str(iter_to_converge))
    print("time taken: " + str(time.time() - start_time))

    img2_new = apply_points_transformation(img2_new, t)

    # save
    np.savetxt(r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\imgs\kitchen\img70_after_icp.csv",
               img2_new, delimiter=",")
