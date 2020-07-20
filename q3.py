import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale, resize, downscale_local_mean, AffineTransform, warp, rotate
from skimage.filters import threshold_otsu, sobel_h, sobel_v, gaussian
from skimage.feature import blob_dog
import argparse
import time

from collections import namedtuple


def rotate_around_pt(img, x0, y0, theta):
    img_rotated = rotate(img, theta, center=(x0, y0))
    return img_rotated


if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 420 A1 Q4')
    parser.add_argument('img1', type=str, help='file for image 1')
    parser.add_argument('x0', type=int, help='x0 for image 1')
    parser.add_argument('y0', type=int, help='y0 for image 1')
    parser.add_argument('theta', type=int, help='theta to rotate image 1')
    parser.add_argument('s', type=float, help='scale to resize image 1')
    args = parser.parse_args()

    # get images
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)

    M = cv2.getRotationMatrix2D((args.x0, args.y0), angle=args.theta, scale=args.s)
    tf_img = cv2.warpAffine(img1, M, img1.shape)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].imshow(tf_img, cmap='gray')
    c = plt.Circle((args.x0, args.y0), 1, color='red', linewidth=1, fill=True)
    ax[0].add_patch(c)
    ax[0].set_title("center=({}, {}) Rotation={} scale={}".format(args.x0, args.y0, args.theta, args.s))
    ax[1].imshow(img1, cmap='gray')
    ax[1].set_title("Original")

    plt.show()

    


