import numpy as np
import cv2
import glob
import re
from matplotlib import pyplot as plt
from skimage.feature import plot_matches, match_descriptors
import argparse
import time

from collections import namedtuple

def matching_cv2(img1, kp1, des1, img2, kp2, des2, plot=False):

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if (plot):
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()

def draw_matches(img1, img2, kp1, kp2, matches, title, x0=None, y0=None, window=None):
    """
    plot the given keypoints and there matches on the given images
    :param img1: first image
    :param img2: second image
    :param kp1: keypoints for the first image
    :param kp2: keypoints for the second image
    :param matches: list of pairs of indices to match the keypoints from the first to second image
    :param title: title for this figure
    :param x0: x point of our ROI for displaying the matches
    :param y0: y point of our ROI for displaying the matches
    :param window: size of the ROI
    :return: None
    """

    filtered_matches = matches

    if ( y0 is not None) and ( x0 is not None) and ( window is not None):
        print(x0, y0, window)
        filtered_matches = []
        for m in matches:
            y, x = kp1[m[0]]
            if ( (x > x0 + window or x < x0 - window ) or (y > y0 + window or y < y0 - window)):
                continue
            else:
                filtered_matches.append(m)

    filtered_matches = np.array(filtered_matches)
    fig, ax = plt.subplots()

    plt.gray()

    plot_matches(ax, img1, img2, kp1, kp2, filtered_matches, only_matches=True, keypoints_color='red')
    ax.set_title(title)

    plt.show()

def mathching_skimage(img1, kp1, des1, img2, kp2, des2, plot=False, x0=None, y0=None, window=None):
    """
    brute force matching using euclidean distance
    :param img1: first image
    :param kp1: keypoints for the first image
    :param des1: descriptors for the first image
    :param img2: second image
    :param kp2: keypoints for the second image
    :param des2: descriptors for the second image
    :param plot: optional enables the function to display the matches
    :param x0: x point of our ROI for displaying the matches
    :param y0: y point of our ROI for displaying the matches
    :param window: size of the ROI
    :return: list of pairs of indices to match the keypoints from the first to second image
    """
    # convert the sift points to np arrays
    orginal_points = np.array([(kp1[idx].pt[1], kp1[idx].pt[0]) for idx in range(len(kp1))], dtype=float)
    prime_points = np.array([(kp2[idx].pt[1], kp2[idx].pt[0]) for idx in range(len(kp2))], dtype=float)
    # main matching step
    matches = match_descriptors(des1, des2, max_ratio=0.8, cross_check=True)

    filtered_matches = matches

    if ( y0 is not None) and ( x0 is not None) and ( window is not None):
        print(x0, y0, window)
        filtered_matches = []
        for m in matches:
            y, x = orginal_points[m[0]]
            if ( (x > x0 + window or x < x0 - window ) or (y > y0 + window or y < y0 - window)):
                continue
            else:
                filtered_matches.append(m)
    if(plot):
        draw_matches(img1, img2, orginal_points, prime_points, filtered_matches, "brute force matches", x0, y0, window)

    return filtered_matches

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

    mathching_skimage(img1, kp1, des1, img2, kp2, des2, True, args.x0, args.y0, args.window)

    


