"""
This code is from Sohail's Assignment 3. But we only started using this code after a3 was submitted, since it was
convenient to reuse some of the functions he had implemented.
----------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
import glob
import re
import argparse

import utils.homography_utils.q8 as q8

from collections import namedtuple

Match = namedtuple('Match', 'x y x_prime y_prime')

def LS_array_from_xi_eq(m):
    return np.array([m.x, m.y, 1, 0, 0, 0, -m.x_prime * m.x, -m.x_prime * m.y, -m.x_prime], dtype=float)

def LS_array_from_yi_eq(m):
    return np.array([ 0, 0, 0, m.x, m.y, 1, -m.y_prime * m.x, -m.y_prime * m.y, -m.y_prime], dtype=float)

def make_homography_LS_matrix(matches):
    """
    use the matches to make the homography liner system matrix
    :param matches: pairs of matched points
    :return: homography liner system matrix
    """

    A_matrix = []

    for m in matches:
        A_matrix.append(LS_array_from_xi_eq(m))
        A_matrix.append(LS_array_from_yi_eq(m))

    return np.array(A_matrix, dtype=float)

def solve_homography_LS_matrix(A_matrix):
    """
    Solve the linear system to find the homography matrix
    :param A_matrix: linear system to solve
    :return: the homography matrix
    """
    eigen_values , eigen_vectors = np.linalg.eig(np.dot(np.transpose(A_matrix), A_matrix))
    h_hat = eigen_vectors[:, np.argmin(eigen_values)]

    # convert the h_hat vector into a matrix before returning it
    return h_hat.reshape(3,3)

def get_transformed_points(keypoints, H_matrix):
    """
    convert the given key points using the given homography matrix
    :param keypoints: list of key points
    :param H_matrix: homography matrix
    :return: list of transformed points
    """
    keypoints_3dim = np.zeros((keypoints.shape[0], keypoints.shape[1] + 1))
    transformed_points = np.zeros(keypoints.shape)

    for i in range(len(keypoints)):
        keypoint_3dim = np.array([[keypoints[i][1], keypoints[i][0], 1]], dtype=float)
        transformed_point_3dim = np.dot(H_matrix, np.transpose(keypoint_3dim))
        normalize_transformed_point = transformed_point_3dim[:2, :] / transformed_point_3dim[2, :]
        new_value = np.transpose(normalize_transformed_point)
        transformed_points[i] = [new_value[0][1], new_value[0][0]]

    return transformed_points

def get_ssd(orginal_points, transformed_points, keypoints_prime):
    ssd = []
    matches = []
    for i in range(len(transformed_points)):
        nearest_keypoints_prime_idx = []
        ssd_for_nearest_keypoints_prime = []
        cur_trans_pt = transformed_points[i]
        for j in range(len(keypoints_prime)):
            if (keypoints_prime[j][0] <= cur_trans_pt[0] + 5 and keypoints_prime[j][1] <= cur_trans_pt[1] + 5):
                nearest_keypoints_prime_idx.append(j)
                ssd_for_nearest_keypoints_prime.append(np.sum((keypoints_prime[j] - cur_trans_pt)**2))
        if (len(ssd_for_nearest_keypoints_prime) == 0):
            continue
        idx_of_min = np.argmin(ssd_for_nearest_keypoints_prime)
        min_ssd = ssd_for_nearest_keypoints_prime[idx_of_min]
        if (min_ssd < 5):
            matches.append((i, nearest_keypoints_prime_idx[idx_of_min]))
            ssd.append(ssd_for_nearest_keypoints_prime[idx_of_min])
    print(len(matches), len(ssd))
    return np.array(matches), np.array(ssd)

def get_ssd_v2(orginal_points, transformed_points, keypoints_prime, binned_array_prime):
    """
    compute the summed squared difference using key points near the transformed points and
    return the inliers
    :param orginal_points: the original key points
    :param transformed_points: transformed points
    :param keypoints_prime: key points to match against
    :param binned_array_prime: key points to match against put into a grid
    :return: list of best inliers and list of the ssd value for those inliers
    """
    ssd = []
    matches = []
    binned_array_prime_h = len(binned_array_prime)
    binned_array_prime_w = len(binned_array_prime[0])
    used_points = []
    for i in range(len(transformed_points)):
        nearest_keypoints_prime_idx = []
        ssd_for_nearest_keypoints_prime = []
        cur_trans_pt = transformed_points[i]
        for y in range(max(0,int(cur_trans_pt[0])-6), min(int(cur_trans_pt[0])+5, binned_array_prime_h)):
            for x in range(max(0,int(cur_trans_pt[1])-6), min(int(cur_trans_pt[1])+5, binned_array_prime_w)):
                for j in range(len(binned_array_prime[y][x])):
                    idx_of_cur_pt_prime = binned_array_prime[y][x][j]
                    nearest_keypoints_prime_idx.append(idx_of_cur_pt_prime)
                    ssd_for_nearest_keypoints_prime.append(np.sum((keypoints_prime[idx_of_cur_pt_prime] - cur_trans_pt)**2))
        if (len(ssd_for_nearest_keypoints_prime) == 0):
            continue
        idx_of_min = np.argmin(ssd_for_nearest_keypoints_prime)
        min_ssd = ssd_for_nearest_keypoints_prime[idx_of_min]
        cur_prime_pt_idx = nearest_keypoints_prime_idx[idx_of_min]
        if (min_ssd < 10 and (not cur_prime_pt_idx in used_points)):
            matches.append((i, cur_prime_pt_idx))
            ssd.append(ssd_for_nearest_keypoints_prime[idx_of_min])
            used_points.append(cur_prime_pt_idx)

    return np.array(matches), np.array(ssd)

def make_binned_array(img_shape, keypoints):
    """
    palace a list of key points into a grid
    :param img_shape: shape of grid
    :param keypoints: list of key points
    :return: grid of key points
    """
    h, w = img_shape
    print(img_shape)
    matrix = [[[] for x in range(w)] for y in range(h)] 
    #grid = np.zeros(img_shape)
    for i in range(len(keypoints)):
        pt = keypoints[i]
        matrix[int(pt[0]) -1][int(pt[1]) -1].append(i)
    print (len(matrix))
    return matrix

def ransac_loop(img1, img2, kp1, kp2, bf_matches, plot=False, x0=None, y0=None, window=None):
    """
    run ransac to get the best homography matrix
    :param img1: first image
    :param img2: second image
    :param kp1: keypoints for the first image
    :param kp2: keypoints for the second image
    :param bf_matches: list of paris of indices that match keypoints form the first image to the second image
    :param plot: option to plot the image
    :param x0: x value of ROI
    :param y0: y value of ROI
    :param window: size of ROI
    :return: homography matrix
    """
    best_base_pairings = []

    orginal_points = np.array([(kp1[idx].pt[1], kp1[idx].pt[0]) for idx in range(len(kp1))], dtype=float)
    keypoints_prime = np.array([(kp2[idx].pt[1], kp2[idx].pt[0]) for idx in range(len(kp2))], dtype=float)
    binned_array_prime = make_binned_array(img2.shape, keypoints_prime)

    number_of_inliers = 0
    best_base_matches = []

    for r in range(100):
        
        base_pairings = []
        match_idxs = np.random.randint(len(bf_matches), size=4)
        # pick 4 pairs of images from the bf_matches
        for i in range(4):
            kp1_idx, kp2_idx = bf_matches[match_idxs[i]]
            xi, yi = kp1[kp1_idx].pt
            xi_prime, yi_prime = kp2[kp2_idx].pt
            base_pairings.append(Match(xi, yi, xi_prime, yi_prime))

        # compute the homography matrix and get the transformed points
        A = make_homography_LS_matrix(base_pairings)
        H_matrix = solve_homography_LS_matrix(A)
        transformed_points = get_transformed_points(orginal_points, H_matrix)

        # use ssd to validate if the transformed points are inliers
        matches, ssd = get_ssd_v2(orginal_points, transformed_points, keypoints_prime, binned_array_prime)
        number_of_cur_inliers = len(matches)
        if(number_of_cur_inliers > number_of_inliers):
            best_base_matches = matches
            number_of_inliers = number_of_cur_inliers
            best_base_pairings = base_pairings

    final_inlier_matches = []
    print("Final accuracy using 4 pts: matches={} ".format(number_of_inliers), best_base_pairings)
    for i in range(number_of_inliers):
        cur_kp1 = orginal_points[best_base_matches[i][0]]
        cur_kp2 = keypoints_prime[best_base_matches[i][1]]
        final_inlier_matches.append(Match(cur_kp1[1], cur_kp1[0], cur_kp2[1], cur_kp2[0]))

    A = make_homography_LS_matrix(final_inlier_matches)
    H_matrix = solve_homography_LS_matrix(A)
    transformed_points = get_transformed_points(orginal_points, H_matrix)
    final_matches, ssd = get_ssd_v2(orginal_points, transformed_points, keypoints_prime, binned_array_prime)
    min_count = len(bf_matches) * 0.5
    print("Final accuracy using {} pts: matches={} in the 50 count ={} ".format(number_of_inliers, len(final_matches), min_count))

    if (plot):
        q8.draw_matches(img1, img2, orginal_points, keypoints_prime, final_matches, "inliers matching points", x0, y0, window)

    return H_matrix, final_matches

if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 420 A4 Q8')

    parser.add_argument('--imgs', default='my_apartment/*.png', type=str, help='Input folder_path\*.png')
    parser.add_argument('img_number', type=int, help='number of the image you want to match')
    parser.add_argument('x0', type=int, help='x0 for image 1')
    parser.add_argument('y0', type=int, help='y0 for image 1')
    parser.add_argument('window', type=int, help='window for image 1')
    args = parser.parse_args()

    print(args.imgs)
    files = glob.glob(args.imgs)
    filenames_in_order = ['' for x in range(len(files))]
    for f in files:
        regex = re.findall(r'\d+', f)
        filenames_in_order[int(regex[-1]) -1] = f

    print(filenames_in_order)

    imgs = []

    # get images
    for f in filenames_in_order:
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    img1, img2 = imgs[args.img_number-1:args.img_number+1]

    # find the keypoints and descriptors with cv2 SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf_matches = q8.mathching_skimage(img1, kp1, des1, img2, kp2, des2, False)
    ransac_loop(img1, img2, kp1, kp2, bf_matches, True, args.x0, args.y0, args.window)






