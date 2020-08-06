import glob
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io


def get_image_set(image_set, depth=False, scale=1):
    """
    Read images given regex representing file name

    :param image_set: paths to images in regex
    :param scale: constant to scale depth down by, if applicable
    :return: list of loaded images
    """
    image_files = glob.glob(image_set)
    loaded_images = []

    # reorder file names to be in numbering order
    filenames_in_order = ['' for _ in range(len(image_files))]
    for f in image_files:
        regex = re.findall(r'\d+', f)
        filenames_in_order[int(regex[-1]) - 1] = f

    # read image files (in order)
    for file in filenames_in_order:
        if depth:
            x = io.imread(file) * scale
        else:
            x = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        loaded_images.append(x)

    return np.stack(loaded_images, axis=0)


def get_rgbd(images, depths, plot=False, scale=1):
    """
    Read RGB images and depth images

    :param images: paths to RGB images in regex. These have to be numbered in format "car1.png", "car2.png",
                   but any initial name (eg. "car" can be substituted for "horse" or "kitchen") and any file
                   extension (eg. ".png" can be substituted for ".jpg" is allowed)
    :param depths: paths to depth images in regex. These have to be numbered in format "car1.png", "car2.png",
                   but any initial name (eg. "car" can be substituted for "horse" or "kitchen") and any file
                   extension (eg. ".png" can be substituted for ".jpg" is allowed)
    :param plot: True to plot these images after they have been read
    :param scale: constant to scale depth down by, if applicable
    :return: list of RGB images, list of depth images
    """
    rgb_images = get_image_set(images)
    depth_images = get_image_set(depths, True, scale)
    if plot:
        fig, axs = plt.subplots(len(rgb_images), 2)
        for i in range(len(rgb_images)):
            axs[i, 0].imshow(rgb_images[i])
            axs[i, 1].imshow(depth_images[i])

        plt.show()

    return rgb_images, depth_images

