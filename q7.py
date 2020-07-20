import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.filters import threshold_otsu, sobel_h, sobel_v, gaussian
from skimage.feature import blob_dog
import argparse
import time

import q1, q2

from collections import namedtuple
Point = namedtuple('Point', 'x y sigma')
Siftpoint = namedtuple('Siftpoint', 'x y sigma n_gradient_m n_gradient_o n_gradient_g_m')
Siftvector = namedtuple('Siftvector', 'x y sigma w_array')


def show_img(img):
    fig, ax = plt.subplots()
    ax.set_title("SIFT opencv")
    ax.imshow(img, cmap='gray')

    plt.show()

def display_histogram(img, siftvectors, x, y, window, width=36):
    """

    :param img:
    :param siftvectors:
    :param x:
    :param y:
    :param window:
    :param width:
    :return:
    """
    bins = [x for x in range(width)]

    for i in range(len(siftvectors)):
        cur_y, cur_x = siftvectors[i].x, siftvectors[i].y
        if ((cur_x > x + window or cur_x < x - window) or (cur_y > y + window or cur_y < y - window)):
            continue

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].bar(bins, siftvectors[i].w_array)
        ax[0].set_title("histogram for keypoint({}, {}) ".format(cur_y, cur_x))
        c = plt.Circle((cur_x, cur_y), 1, color='red', linewidth=1, fill=True)
        ax[1].add_patch(c)
        ax[1].imshow(img, cmap='gray')

    plt.show()

def display_histogram_cv2(img, keypoints, all_des, x, y, window):
    """
    display the cv2 histogram for a certain region
    :param img: image to plot
    :param keypoints: key points for current image
    :param all_des: descriptors
    :param x: x value of ROI
    :param y: y value of ROI
    :param window: size of ROI
    :return: None
    """
    bins = [i for i in range(128)]
    keypoints_np = np.array([(keypoints[idx].pt[1], keypoints[idx].pt[0]) for idx in range(len(keypoints))], dtype=float)
    print("cv2 kp=",len(keypoints_np))
    for i in range(len(keypoints_np)):
        cur_y, cur_x = keypoints_np[i]
        if ((cur_x > x + window or cur_x < x - window) or (cur_y > y + window or cur_y < y - window)):
            continue

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].bar(bins, all_des[i])
        ax[0].set_title("histogram for keypoint({}, {}) ".format(cur_y, cur_x))
        c = plt.Circle((cur_x, cur_y), 1, color='red', linewidth=1, fill=True)
        ax[1].add_patch(c)
        ax[1].imshow(img, cmap='gray')

    plt.show()

if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 420 A1 Q4')
    parser.add_argument('img1', type=str, help='file for image 1')
    parser.add_argument('x', type=int, help='x value of the center roi')
    parser.add_argument('y', type=int, help='y value of the center roi')
    parser.add_argument('window', type=int, help='window for roi')
    args = parser.parse_args()

    # get images
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    padded_img = np.pad(img1, ((0,224), (0,424)))
    print(padded_img.shape)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, all_des = sift.detectAndCompute(img1, None)
    img = cv2.drawKeypoints(img1, keypoints, img1)
    show_img(img)
    siftvectors = q2.get_siftvectors(padded_img, neighborhood_size=3)
    q2.plot_keypoints(siftvectors, img1)
    display_histogram_cv2(img1, keypoints, all_des, args.x, args.y, args.window)
    display_histogram(img1, siftvectors, args.x, args.y, args.window)



