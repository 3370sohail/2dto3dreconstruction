import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale, resize, downscale_local_mean, AffineTransform, warp, rotate
from skimage.filters import threshold_otsu, sobel_h, sobel_v, gaussian
from skimage.feature import blob_dog, plot_matches, match_descriptors
import argparse
import time

from collections import namedtuple
Point = namedtuple('Point', 'x y sigma')
Siftpoint = namedtuple('Siftpoint', 'x y sigma n_gradient_m n_gradient_o n_gradient_g_m')
Siftvector = namedtuple('Siftvector', 'x y sigma w_array')

import q2

def get_kp_des(img, x0, y0, type='b'):
    """
    get 2 parallel list of key points and descriptors for the given image.
    if the type is b we are using the Bhattacharyya coefficient and must normalize the histogram.
    :param img: image to process
    :param x0: x point of our ROI
    :param y0: y point of our ROI
    :param type: used to determine of we want to normalize the histogram for the Bhattacharyya coefficient
    :return: 2 list one with the keypoint and one with the descriptors
    """
    siftvectors = q2.get_siftvectors(img)

    x_y_points = []
    des = []
    for vec in siftvectors:
        x, y = vec.y * 2**vec.sigma, vec.x * 2**vec.sigma
        if ( (x > x0 + 200 or x < x0 - 200 ) or (y > y0 + 200 or y < y0 - 200)):
            continue
        x_y_points.append((vec.x * 2**vec.sigma, vec.y * 2**vec.sigma))
        if(type == 'b'):
            # normalize all the histograms
            des.append(vec.w_array / np.sum(vec.w_array))
        else:
            des.append(vec.w_array)
    print(len(x_y_points), np.sum(des))
    return np.array(x_y_points), np.array(des)

def draw_matches(img1, img2, kp1, kp2, des1, des2, matches):
    """
    plot the given keypoints and there matches on the given images
    :param img1: first image
    :param img2: second image
    :param kp1: keypoints for the first image
    :param kp2: keypoints for the second image
    :param des1: descriptors for the first image
    :param des2: descriptors for the second image
    :param matches: list of paris of indices that match keypoints form the first image to the second image
    :return: None
    """
    fig, ax = plt.subplots()

    plt.gray()

    plot_matches(ax, img1, img2, kp1, kp2, matches,only_matches=False,keypoints_color='red')
    ax.set_title("sift feature matches")

    plt.show()

def get_matches_cross_check(des1, des2, cross=True):
    """
    match 2 set of descriptors

    adapted the code for cross checking and ratio test from
    https://github.com/scikit-image/scikit-image/blob/v0.17.2/skimage/feature/match.py#L5

    :param des1: list of descriptors from the first  image
    :param des2: list of descriptors from the second image
    :param cross: enables cross checking
    :return:
    """
    b_c_values = []
    matches = []
    for i in range(len(des1)):
        cur_des = des1[i]
        repated_cur_des = [cur_des for c in range(len(des2))]
        b_c = np.sum(np.sqrt(repated_cur_des * des2), axis=1)
        b_c_values.append(b_c)
        max_idx = np.argmax(b_c)
        matches.append((i, max_idx))

    b_c_values = np.array(b_c_values)

    print(b_c_values.shape)

    des1_idxs = np.arange(des1.shape[0])
    des2_idxs = np.argmax(b_c_values, axis=1)

    #cross checking
    if (cross):
        matches1 = np.argmax(b_c_values, axis=0)
        mask = des1_idxs == matches1[des2_idxs]
        des1_idxs = des1_idxs[mask]
        des2_idxs = des2_idxs[mask]

    best_distances = b_c_values[des1_idxs, des2_idxs]
    glob_max = np.max(b_c_values)
    print('best largest max', glob_max)

    # threshold the matches based on the how good they are
    mask = best_distances > glob_max * 0.5
    des1_idxs = des1_idxs[mask]
    des2_idxs = des2_idxs[mask]

    print(len(matches), len(des2))

    return np.column_stack((des1_idxs, des2_idxs))


if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 4200 A1 Q4')
    parser.add_argument('img1', type=str, help='file for image 1')
    parser.add_argument('x0', type=int, help='x0 for image 1')
    parser.add_argument('y0', type=int, help='y0 for image 1')
    parser.add_argument('theta', type=int, help='theta to rotate image 1')
    parser.add_argument('s', type=float, help='scale to resize image 1')
    parser.add_argument('type', type=str, help='scale to resize image 1')
    args = parser.parse_args()

    # get images
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)

    # transform the image
    M = cv2.getRotationMatrix2D((args.x0, args.y0), angle=args.theta, scale=args.s)
    tf_img = cv2.warpAffine(img1, M, img1.shape)

    kp1, des1 = get_kp_des(img1, args.x0, args.y0, type=args.type)
    kp2, des2 = get_kp_des(tf_img, args.x0, args.y0, type=args.type)

    if (args.type == 'b'):
            matches = get_matches_cross_check(des1, des2)
    else:
        matches = match_descriptors(des1, des2, cross_check=True)
    draw_matches(img1, tf_img, kp1, kp2, des1, des2, matches)





