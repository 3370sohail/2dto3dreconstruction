import csv
import os

import cv2
import numpy as np
import open3d as o3d
from scipy import optimize
from skimage import io

from utils.icp import icp


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

    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)

    return kp1, kp2, matches


def refine_matches(matches):
    """
    Right now we are just taking the best 10 matches

    :param matches: list of matches
    :return: refined list of matches
    """
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:10]
    return matches


def pts_to_arr(pts, shape):
    """
    Write points to image with shape shape

    :param pts: n x 3 ndarray, where the 3 is x y z coordinates
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


def register_imgs(img1_rgb, img2_rgb, img1_depth, img2_depth, scale=1., filter_pts_frac=1., partial_set_frac=1.,
                  img1_pts=None, img2_pts=None):
    """
    Perform global image registration given the RGB and depth of two images by

        1. Performing global registration by extracting keypoints from RGB images
        2. Performing local registration by ICP

    :param img1_rgb: (h, w, 3) image
    :param img2_rgb: (h, w, 3) image
    :param img1_depth: (h, w) depth image
    :param img2_depth: (h, w) depth image
    :param scale: scaling factor for loading point clouds (see in-depth explanation in depth_to_voxel() function)
    :param filter_pts_frac: fraction of all the points to use when performing ICP. Must be in range (0, 1], though
                            choosing a good value would depend on the density of the depth we are working with.
    :param partial_set_frac: fraction of expected overlap between the two images when performing ICP. Must be in
                             range (0, 1]. In the case of video data, this value should usually be close to 1,
                             though maybe not exactly 1, since we can expect a high overlap between frames of a
                             video. Alternatively, if we are taking one frame from every couple of frames, this
                             value should be slightly lower.
    :param img1_pts: (h x w, 3) points, optional. If this is given, scale is not needed for image 1.
    :param img2_pts: (h x w, 3) points, optional. If this is given, scale is not needed for image 1.
    :return: image 1 point cloud,
             image 2 point cloud,
             4x4 transformation matrix that maps image 1 to image 2
    """
    # convert depth to point cloud
    if img1_pts is None:
        img1_pts = depth_to_voxel(img1_depth, scale=scale)
    if img2_pts is None:
        img2_pts = depth_to_voxel(img2_depth, scale=scale)

    # find RGB matches
    kp1, kp2, matches = generate_keypoints_and_match(img1_rgb, img2_rgb)
    matches = refine_matches(matches)

    # # draw matches
    # img = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches, outImg=None,
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
    img2_new = get_transformed_points(img1_pts, h)

    # filter points (since we have a dense point cloud)
    pts_idx1 = np.random.choice(img2_new.shape[0], int(img2_new.shape[0] * filter_pts_frac), replace=False)
    pts_idx2 = np.random.choice(img2_pts.shape[0], int(img2_pts.shape[0] * filter_pts_frac), replace=False)

    # ICP for fine registration
    t, _, iter_to_converge = icp(img2_new[pts_idx1], img2_pts[pts_idx2],
                                 tolerance=0.001,
                                 max_iterations=100,
                                 partial_set_size=partial_set_frac)

    # return point clouds and transformation
    return img1_pts, img2_pts, np.dot(h, t)
