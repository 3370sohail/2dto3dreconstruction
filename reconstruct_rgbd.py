import argparse

import reconstruct as rec
import utils.rgbd as rgbd

if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='Depth Point Cloud Registration and 3D Model Reconstruction')

    parser.add_argument('--rgb', default='./image_sets/sofa4/*.jpg', type=str,
                        help='Input filename or folder for RGB images')
    parser.add_argument('--depth', default='./image_sets/sofa4/*.png', type=str,
                        help='Input filename or folder for depth images')
    parser.add_argument('--inter', default='true', type=bool,
                        help='Read point clouds')

    parser.add_argument('--mode', default='fpfh', type=str, choices=['fpfh', 'rigid3d', '3dhomo'],
                        help='Global registration method')
    parser.add_argument('--voxel', default=20, type=int,
                        help='Size of voxel to downsample for FPFH. Do not use if not using FPFH for mode option.')
    parser.add_argument('--fast', action="store_true",
                        help='Enable to use fast global registration for FPFH. Do not use if not using FPFH for '
                             'mode option')

    parser.add_argument('--surface', default='poisson', type=str, choices=['poisson', 'ball_point'],
                        help='Method of generating surface mesh')

    parser.add_argument('--save_intermediate', action='store_true',
                        help='Enable to store intermediate results (in out_folder)')
    parser.add_argument('--out_folder', default='./image_sets/sofa4', type=str,
                        help='Path to folder to save generated point clouds and meshes in')
    parser.add_argument('--out_name', default='car-d', type=str,
                        help='Name of image set to save as')

    parser.add_argument('--plot', action='store_true',
                        help='Enable to plot intermediate results in pipeline')

    args = parser.parse_args()

    depth_scale = 0.1
    if args.inter:
        depth_scale = 1

    rgb_images, depth_images = rgbd.get_rgbd(args.rgb, args.depth, args.plot, depth_scale)

    if args.inter:
        point_clouds = rec.depth_images_to_3d_pts_ld(depth_images)
    else:
        point_clouds = rec.depth_images_to_3d_pts(depth_images, scale=depth_scale)

    poisson = args.surface == "poisson"

    if args.mode == "fpfh":
        # fast voxel = 10 ransac voxel = 20
        rec.fpfh(point_clouds,
                 voxel_ds_size=args.voxel,
                 fast=args.fast,
                 save_intermediate=args.save_intermediate,
                 out_folder=args.out_folder,
                 image_set_name=args.out_name,
                 poisson=poisson,
                 plot=args.plot)

    elif args.mode == "rigid3d":
        np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img = rec.get_kps_decs(rgb_images)
        rec.rigid3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img,
                         save_intermediate=args.save_intermediate,
                         out_folder=args.out_folder,
                         image_set_name=args.out_name,
                         poisson=poisson,
                         plot=args.plot)

    elif args.mode == "3dhomo":
        rec.trans3d_proc(point_clouds, rgb_images, depth_images,
                         save_intermediate=args.save_intermediate,
                         out_folder=args.out_folder,
                         image_set_name=args.out_name,
                         poisson=poisson,
                         plot=args.plot)

    else:
        raise Exception("Not a valid reconstruction method!")
