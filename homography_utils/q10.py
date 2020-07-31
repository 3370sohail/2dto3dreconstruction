import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import argparse
import time
import re

import q8, q9

from collections import namedtuple

Match = namedtuple('Match', 'x y x_prime y_prime')

Orginal_image_shape = (0, 0)


def get_tranfromed_imgs(img1, img2, shape):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf_matches = q8.mathching_skimage(img1, kp1, des1, img2, kp2, des2)
    H_matrix = q9.ransac_loop(img1, img2, kp1, kp2, bf_matches)

    new_img = cv2.warpPerspective(img1, H_matrix, (shape[1], shape[0])).astype(np.uint8)

    merged_img = np.maximum(new_img, img2)
    return merged_img

def pad_for_current_level(img_list, right=False):
    """
    pad the images using the original width of the base image
    :param img_list: images to pad
    :param right: what side we should pad
    :return:
    """
    rows, cols = Orginal_image_shape
    new_img_list = []
    for p in img_list:
        if (right):
            new_img_list.append(np.pad(p, ((0, 100), (0, cols))))
        else:
            new_img_list.append(np.pad(p, ((0, 100), (cols, 0))))

    return new_img_list


def recur_stitch(imgs, plot=False, right=False):
    """
    stitches two successive images one after another
    :param imgs: images to stitch
    :param plot: option to display the stitching process
    :param right: what side we should be snitching by
    :return: return the final stitched image
    """

    new_imgs = pad_for_current_level(imgs, right=right)

    for i in range(1, len(imgs)):
        if (right):
            t_img = get_tranfromed_imgs(new_imgs[i], new_imgs[i - 1], new_imgs[0].shape)
        else:
            t_img = get_tranfromed_imgs(new_imgs[i - 1], new_imgs[i], new_imgs[0].shape)
        new_imgs[i] = t_img
        new_imgs = pad_for_current_level(new_imgs, right=right)

    if (plot):
        fig, ax = plt.subplots()

        ax.imshow(new_imgs[len(imgs) - 1], cmap='gray')

        plt.show()

    return new_imgs[len(imgs) - 1]


def pad_same_width(img, right=False):
    """
    pad the left or right side of the image using the size of the
    :param img: images to stitch
    :param right: what side we should pad
    :return: padded image
    """
    rows, cols = img.shape
    if (right):
        return np.pad(img, ((0, 0), (cols, 0))).astype(np.uint8)

    return np.pad(img, ((0, 0), (0, cols))).astype(np.uint8)


def crop_img(img):
    """
    crop the given image such that it moved any empty space
    :param img: img to crop
    :return: cropped image
    """
    min_idx_x = np.inf
    max_idx_x = 0
    min_idx_y = np.inf
    max_idx_y = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j] != 0):
                if (j > max_idx_x):
                    max_idx_x = j
                if (j < min_idx_x):
                    min_idx_x = j
                if (i > max_idx_y):
                    max_idx_y = i
                if (i < min_idx_y):
                    min_idx_y = i

    return img[min_idx_y:max_idx_y + 1, min_idx_x:max_idx_x + 1]


def final_stitch(left, right):
    """
    stitch the 2 side together
    :param left: left image
    :param right: right image
    :return: final image
    """
    rows, cols = Orginal_image_shape
    # trim the images so that we are only matching the part of the left and right images that aren't to warped
    trimmed_left = np.zeros(left.shape)
    trimmed_left[:, left.shape[1] - cols:] = left[:, left.shape[1] - cols:]
    trimmed_right = np.zeros(right.shape)
    trimmed_right[:, :cols] = right[:, :cols]
    trimmed_right = pad_same_width(trimmed_right, right=True)
    trimmed_left = pad_same_width(trimmed_left)
    left = pad_same_width(left)
    right = pad_same_width(right, right=True)

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(trimmed_right, None)
    kp2, des2 = sift.detectAndCompute(trimmed_left, None)

    bf_matches = q8.mathching_skimage(trimmed_right, kp1, des1, trimmed_left, kp2, des2)
    H_matrix = q9.ransac_loop(trimmed_right, trimmed_left, kp1, kp2, bf_matches)

    new_img = cv2.warpPerspective(right, H_matrix, (right.shape[1], right.shape[0])).astype(np.uint8)
    merged_img = np.maximum(new_img, left)

    # crop and plot the image
    show_img(crop_img(merged_img))

    return merged_img


def show_img(img):
    """
    just plot the image
    :param img: image to plot
    :return: none
    """
    fig, ax = plt.subplots()
    ax.set_title("final stitched img")
    ax.imshow(img, cmap='gray')

    plt.show()


if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 420 A1 Q4')

    parser.add_argument('--imgs', default='cars2/*.jpg', type=str, help='Input folder_path\*.png')
    args = parser.parse_args()

    print(args.imgs)
    files = glob.glob(args.imgs)
    print(files)
    filenames_in_order = ['' for x in range(len(files))]
    for f in files:
        regex = re.findall(r'\d+', f)
        filenames_in_order[int(regex[-1]) - 1] = f

    print(filenames_in_order)

    imgs = []
    num_imgs = len(filenames_in_order)

    if (num_imgs == 1):
        show_img(imgs[0])

    for i in range(num_imgs):
        cur_img = cv2.imread(filenames_in_order[i], cv2.IMREAD_GRAYSCALE)
        imgs.append(cur_img)

    print("num imgs", len(imgs))

    Orginal_image_shape = imgs[0].shape

    half_way_index = num_imgs // 2

    left = recur_stitch(imgs, plot=True, right=False)

    # based on if the image set has an even or odd number of images palace the matching in left and right stitching







