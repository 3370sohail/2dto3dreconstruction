import os
from scipy import optimize

import cv2
import h5py
import numpy as np


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
    matches = matches[:4]

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


def LS_array_from_xi_eq(x, y, z):
    return np.array([x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0])


def LS_array_from_yi_eq(x, y, z):
    return np.array([0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0])


def LS_array_from_zi_eq(x, y, z):
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1])


def make_homography_LS_matrix(img1_kp):
    """
    use the matches to make the homography liner system matrix
    :param matches: pairs of matched points
    :return: homography liner system matrix
    """
    a_matrix = []

    for i in range(len(img1_kp)):
        a_matrix.append(LS_array_from_xi_eq(img1_kp[i][0], img1_kp[i][1], img1_kp[i][2]))
        a_matrix.append(LS_array_from_yi_eq(img1_kp[i][0], img1_kp[i][1], img1_kp[i][2]))
        a_matrix.append(LS_array_from_zi_eq(img1_kp[i][0], img1_kp[i][1], img1_kp[i][2]))

    a_matrix = np.array(a_matrix, dtype=np.float32)
    return a_matrix


def solve_homography_LS_matrix(A_matrix):
    """
    Solve the linear system to find the homography matrix
    :param A_matrix: linear system to solve
    :return: the homography matrix
    """
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.transpose(A_matrix), A_matrix))
    h_hat = eigen_vectors[:, np.argmin(eigen_values)]

    # convert the h_hat vector into a matrix before returning it
    return h_hat.reshape(4, 4)


def homo_rigid_transform_3D(img1_kp, img2_kp):
    A = make_homography_LS_matrix(img1_kp)
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


if __name__ == "__main__":
    # folder = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\aunt_jemima_original_syrup"
    #
    # # load images and depth
    # depth = []
    # imgs = []
    # for i in range(0, 100, 3):
    #
    #     fname = os.path.join(folder, f"NP1_{str(i)}.h5")
    #     with h5py.File(fname, "r") as f:
    #         depth.append(f.get('depth')[:])
    #
    #     fname = fname.replace("h5", "jpg")
    #     imgs.append(cv2.imread(fname))
    #
    # depth = np.array(depth)
    # imgs = np.array(imgs)

    # load images
    img1 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car1.jpg"
    img2 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car2.jpg"
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # load point clouds
    img1_pts = np.genfromtxt(
        r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_0.csv", delimiter=",")
    img2_pts = np.genfromtxt(
        r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_1.csv", delimiter=",")
    img1_depth = pts_to_arr(img1_pts, (img1.shape[0] // 2, img1.shape[1] // 2))
    img2_depth = pts_to_arr(img2_pts, (img2.shape[0] // 2, img2.shape[1] // 2))
    img1_depth = img1_depth.repeat(2, axis=0).repeat(2, axis=1)
    img2_depth = img2_depth.repeat(2, axis=0).repeat(2, axis=1)

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
    h = homo_rigid_transform_3D(img1_kp, img2_kp)

    # apply transformation
    img2_new = get_transformed_points(img1_pts, h)

    # save
    np.savetxt(r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_0_rigid3d.csv",
               img2_new, delimiter=",")

