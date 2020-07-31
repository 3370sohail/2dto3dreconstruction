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

import q1

from collections import namedtuple
Point = namedtuple('Point', 'x y sigma')
Siftpoint = namedtuple('Siftpoint', 'x y sigma n_gradient_m n_gradient_o n_gradient_g_m')
Siftvector = namedtuple('Siftvector', 'x y sigma w_array or_hist')


def get_gradient_data(keypoints, dx_of_image_pyramid, dy_of_image_pyramid):

    gradient_data = []
    gradient_mags = []
    gradient_ors = []

    for n in range(len(dy_of_image_pyramid)):
        mags = np.sqrt(np.square(dx_of_image_pyramid[n]) + np.square(dy_of_image_pyramid[n]))
        gradient_mags.append(np.pad(mags, ((9, 9), (9, 9)), 'constant'))
        orintation = np.degrees(np.arctan2(dy_of_image_pyramid[n], dx_of_image_pyramid[n]))
        for i in range(orintation.shape[0]):
            for j in range(orintation.shape[1]):
                cur_value = orintation[i,j]
                if ( (not math.isnan(cur_value)) and cur_value < 0):
                    orintation[i,j] = 360 + cur_value
        gradient_ors.append(np.pad(orintation, ((9, 9), (9, 9)), 'constant'))

    gussian = q1.create_2d_gaussian_matrix(4, 17)
    gussian_trim = gussian[0:-1,0:-1]
    #print(gussian_trim.shape, "gussian : \n", gussian_trim)

    for p in keypoints:
        n_gradient_mag = gradient_mags[p.sigma][p.x+9-8:p.x+9+8, p.y+9-8:p.y+9+8]
        n_gradient_or = gradient_ors[p.sigma][p.x+9-8:p.x+9+8, p.y+9-8:p.y+9+8]
        n_gradient_mag_w = n_gradient_mag * gussian_trim
        gradient_data.append(Siftpoint(p.x, p.y, p.sigma, n_gradient_mag, n_gradient_or, n_gradient_mag_w))

    return gradient_data

def reorgnize(list_to_reorder, index):
    new_list = []
    for i in range(index, -1, -1):
        new_list.append(list_to_reorder[i])
    for i in range(len(list_to_reorder)-1, index, -1):
        new_list.append(list_to_reorder[i])
    return new_list
    
def put_into_bins(keypoints):
    
    bins = [x for x in range(0,370,10)]
    Siftvectors = []
    print(bins, len(bins))
    peaks_list = []
    count = 0
    for p in keypoints:
        bin_sum_values = np.zeros(len(bins))
        found_bin = np.digitize(p.n_gradient_o, bins, right=False)
        for i in  range(found_bin.shape[0]):
            for j in  range(found_bin.shape[1]):
                if (found_bin[i, j] == 0):
                    print(p.n_gradient_o[i, j])
                else:
                    bin_sum_values[found_bin[i, j] -1] += p.n_gradient_g_m[i, j]

        if(np.sum(bin_sum_values) == 0):
            continue
        peaks = signal.find_peaks_cwt(bin_sum_values, [1]) # noise_perc=0.1

        # max value in bins
        best_peak_idx = None
        best_peak = 0
        for k in peaks:
            if(best_peak < bin_sum_values[k]):
                best_peak_idx = k
                best_peak = bin_sum_values[k]
        best_peak_idx = np.argmax(bin_sum_values)
        reordered = reorgnize(bin_sum_values[:-1], best_peak_idx)
        peaks_list.append(peaks)
        Siftvectors.append(Siftvector(p.x, p.y, p.sigma, reordered, bin_sum_values[:-1]))
        count += 1

    return Siftvectors

def display_histogram(siftvectors, pt_idx, width=36):
    bins = [x for x in range(width)]
    print(siftvectors[pt_idx].w_array)
    kp = siftvectors[pt_idx]

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].bar(bins, kp.or_hist)
    ax[0].set_title("orientation histogram for keypoint({}, {}) ".format(kp.y * 2 ** kp.sigma, kp.x * 2 ** kp.sigma))
    ax[1].bar(bins, kp.w_array)
    ax[1].set_title("shifted histogram for keypoint({}, {}) ".format(kp.y * 2 ** kp.sigma, kp.x * 2 ** kp.sigma))

    plt.show()


def get_siftvectors(img, img_pyramid_size=7, neighborhood_size=5):
    res = q1.image_pyramid(img, img_pyramid_size)
    res_DoG = q1.DoG(res)

    keypoints = q1.find_keypoints(res_DoG, neighborhood_size=neighborhood_size)
    keypoints = q1.filter_edge_keypoints(res_DoG, keypoints)

    dx_of_image_pyramid, dy_of_image_pyramid = q1.get_dx_dy_of_image_pyramid(res)

    gradient_data = get_gradient_data(keypoints, dx_of_image_pyramid, dy_of_image_pyramid)

    return put_into_bins(gradient_data)

def plot_keypoints(keypoints, img1 ):
    """
    plots a set of keypoint on to an image using matplotlib
    :param keypoints: list of tuples of type Point to plot
    :param img1: the image to plot the points on
    :return: None
    """
    fig,ax = plt.subplots(1)
    plt.imshow(img1, cmap='gray')
    plt.title("key-points ")

    colors = ["blue", "green", "yellow", "magenta", "red"]

    for kp in keypoints:
        c = plt.Circle((kp.y * 2 ** kp.sigma, kp.x * 2 ** kp.sigma), 1, color=colors[kp.sigma], linewidth=1, fill=True)
        ax.add_patch(c)

    plt.show()

if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 420 A1 Q4')
    parser.add_argument('img1', type=str, help='file for image 1')
    parser.add_argument('idx', type=int, help='idx of the point you want to display')
    args = parser.parse_args()

    # get images
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    
    siftvectors = get_siftvectors(img1)

    display_histogram(siftvectors, args.idx)
    plot_keypoints(siftvectors, img1)



