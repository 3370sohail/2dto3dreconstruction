
import csv
import time

import os

import cv2
import numpy as np
import glob
import open3d as o3d
from scipy import optimize
from skimage import io
import re
from matplotlib import pyplot as plt

def get_image_set(image_set, depth=False, scale=1):
    image_files = glob.glob(image_set)


    loaded_images = []
    print(image_files)

    filenames_in_order = ['' for x in range(len(image_files))]
    for f in image_files:
        regex = re.findall(r'\d+', f)
        filenames_in_order[int(regex[-1]) -1] = f

    print(filenames_in_order)
    for file in filenames_in_order:
        if depth:
            x = io.imread(file) * scale
        else:
            x = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        print(x.shape)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def get_rgbd(images, depths, plot=False, scale=1):
    rgb_images = get_image_set(images)
    depth_images = get_image_set(depths, True, scale)
    if plot:
        fig, axs = plt.subplots(len(rgb_images), 2)
        for i in range(len(rgb_images)):
            axs[i, 0].imshow(rgb_images[i])
            axs[i, 1].imshow(depth_images[i])

        plt.show()

    return rgb_images, depth_images

