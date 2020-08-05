import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d

import argparse

import utils.rgbd as rgbd
import utils.utils as utils
import open3d_utils.fpfh as o3d_utils
import homography_utils.q8 as q8
import homography_utils.q9 as q9
import homo3d
import reconstruct as rec


def fpfh(point_clouds, dump=False, dump_folder=None, image_set_name=None, poisson=True, plot=True):
    pcds = rec.make_pcds(point_clouds, dump, dump_folder, image_set_name)
    all_results = []
    for i in range(1, len(pcds)):

        src_pcd, src_fpfh = o3d_utils.preprocess_point_cloud(pcds[i], 10, 20, 30, 50, 100, plot)
        tar_pcd, tar_fpfh = o3d_utils.preprocess_point_cloud(pcds[i - 1], 10, 20, 30, 50, 100, plot)

        # results = o3d_utils.execute_fast_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 15)
        results = o3d_utils.execute_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 10)
        if plot:
            o3d_utils.visualize_transformation(src_pcd, tar_pcd, results.transformation)
        print(results)
        print(results.transformation)
        all_results.append(results.transformation)

    rec.chain_transformation(pcds, all_results, dump, dump_folder, image_set_name, poisson, plot)


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')

    parser.add_argument('--input', default='./image_sets/car-d/*.jpg', type=str, help='Input filename or folder.')
    parser.add_argument('--depth', default='./image_sets/car-d/*.png', type=str, help='Trained Keras model file.')
    parser.add_argument('--mode', default='fpfh', type=str, help='method of reconstruction')
    parser.add_argument('--surface', default='ball', type=str, help='method of reconstruction')
    parser.add_argument('--dump', default='yes', type=str, help='method of reconstruction')
    parser.add_argument('--folder', default='./image_sets/car-d', type=str, help='method of reconstruction')
    parser.add_argument('--name', default='car-d', type=str, help='method of reconstruction')
    parser.add_argument('--plot', default='no', type=str, help='method of reconstruction')
    args = parser.parse_args()

    plot = False
    if args.plot == "yes":
        plot = True

    rgb_images, depth_images = rgbd.get_rgbd(args.input, args.depth, plot)

    poisson = False
    if args.surface == "poisson":
        poisson = True

    dump = False
    if args.dump == "yes":
        dump = True

    np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img = rec.get_kps_decs(rgb_images)
    point_clouds = rec.depth_images_to_3d_pts(depth_images)
    if (args.mode == "fpfh"):
        fpfh(point_clouds, dump, args.folder, args.name, poisson, plot)
    elif (args.mode == "rigid3d"):
        rec.rigid3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump,
                         args.folder, args.name, poisson, plot)
    else:
        rec.homo3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump,
                        args.folder, args.name, poisson, plot)
