import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import glob
import re
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale, resize, downscale_local_mean, AffineTransform, warp, rotate, ProjectiveTransform
from skimage.filters import threshold_otsu, sobel_h, sobel_v, gaussian
from skimage.feature import blob_dog, plot_matches, corner_harris, corner_subpix, corner_peaks
from scipy.spatial import distance
import open3d as o3d

import argparse
import time

import dense_depth.depth as dd
import utils
import open3d_utils.fpfh as o3d_utils

def make_pcds(point_clouds, dump=False, dump_folder=None, image_set_name=None):
    """

    Args:
        point_clouds:
        dump:
        dump_folder:
        image_set_name:

    Returns:

    """
    pcds = []
    for i in range(len(point_clouds)):
        if dump:
            utils.voxel_to_csv(point_clouds[i], '{}/{}_{}.csv'.format(dump_folder, image_set_name, i))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds[i])
        pcds.append(pcd)

    return pcds

def fpfh(point_clouds):

    pcds = make_pcds(point_clouds)
    for i in range(1, len(pcds)):

        src_pcd, src_fpfh = o3d_utils.preprocess_point_cloud(pcds[i-1], 0.05, 30, 0.05, 100)
        tar_pcd, tar_fpfh = o3d_utils.preprocess_point_cloud(pcds[i], 0.05, 30, 0.05, 100)

        results = o3d_utils.execute_fast_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 1)
        o3d_utils.visualize_transformation(pcds[i], pcds[i-1], results.transformation)

def depth_images_to_3d_pts(depth_images):
    return [utils.depth_to_voxel(img) for img in depth_images]


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='./models/kitti.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='./image_sets/cars/*.jpg', type=str, help='Input filename or folder.')
    args = parser.parse_args()

    rgb_images, depth_images = dd.get_depth(args.model, args.input)

    point_clouds = depth_images_to_3d_pts(depth_images)

    fpfh(point_clouds)

