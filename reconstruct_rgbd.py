import argparse

import numpy as np

import open3d_utils.fpfh as o3d_utils
import reconstruct as rec
import utils.rgbd as rgbd


def fpfh(point_clouds, voxel=10, fast=False, dump=False, dump_folder=None, image_set_name=None, poisson=True, plot=True):
    pcds = rec.make_pcds(point_clouds, dump, dump_folder, image_set_name)
    all_results = []
    for i in range(1, len(pcds)):
        norm_radius = voxel * 2
        fpfh_radius = voxel * 5
        src_pcd, src_fpfh = o3d_utils.preprocess_point_cloud(pcds[i], voxel, norm_radius, 30, fpfh_radius, 100, plot)
        tar_pcd, tar_fpfh = o3d_utils.preprocess_point_cloud(pcds[i - 1], voxel, norm_radius, 30, fpfh_radius, 100, plot)

        if fast:
            results = o3d_utils.execute_fast_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, voxel)
        else:
            results = o3d_utils.execute_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, voxel)

        if plot:
            o3d_utils.visualize_transformation(src_pcd, tar_pcd, results.transformation)
        print(results)
        print(results.transformation)
        all_results.append(results.transformation)

    rec.chain_transformation(pcds, all_results, dump, dump_folder, image_set_name, poisson, plot)


def depth_largobject(depth_img):
    img_shape = depth_img.shape
    homo_cords = np.zeros((img_shape[0] * img_shape[1], 3))
    f_x = 525
    f_y = 525
    c_x = 319.5
    c_y = 239.5

    count = 0
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            z = depth_img[i, j]
            x = (j + 1 - c_x) * z / f_x
            y = (i + 1 - c_y) * z / f_y
            homo_cords[count] = [x, y, z]
            count = count + 1

    return homo_cords


def depth_images_to_3d_pts_ld(depth_images):
    return [depth_largobject(img) for img in depth_images]
    #return [utils.depth_to_voxel_ld(img) for img in depth_images]


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')

    parser.add_argument('--input', default='./image_sets/car-d/*.jpg', type=str, help='Input filename or folder.')
    parser.add_argument('--depth', default='./image_sets/car-d/*.png', type=str, help='Trained Keras model file.')
    parser.add_argument('--mode', default='fpfh', type=str, help='method of reconstruction')
    parser.add_argument('--voxel', default=10, type=int, help='method of reconstruction')
    parser.add_argument('--inter', default='no', type=str, help='method of reconstruction')
    parser.add_argument('--fast', default='no', type=str, help='method of reconstruction')
    parser.add_argument('--surface', default='ball', type=str, help='method of reconstruction')
    parser.add_argument('--dump', default='yes', type=str, help='method of reconstruction')
    parser.add_argument('--folder', default='./image_sets/car-d', type=str, help='method of reconstruction')
    parser.add_argument('--name', default='car-d', type=str, help='method of reconstruction')
    parser.add_argument('--plot', default='no', type=str, help='method of reconstruction')
    args = parser.parse_args()

    depth_scale = 0.7
    if args.inter == 'yes':
        depth_scale = 1

    plot = False
    if args.plot == "yes":
        plot = True

    rgb_images, depth_images = rgbd.get_rgbd(args.input, args.depth, plot, depth_scale)

    poisson = False
    if args.surface == "poisson":
        poisson = True

    dump = False
    if args.dump == "yes":
        dump = True

    if args.inter == 'yes':
        point_clouds = depth_images_to_3d_pts_ld(depth_images)
    else:
        point_clouds = rec.depth_images_to_3d_pts(depth_images, scale=depth_scale)

    fast = False
    if args.fast == 'yes':
        fast = True

    if args.mode == "fpfh":
        # fast voxel = 10 ransac voxel = 20
        fpfh(point_clouds, args.voxel, fast, dump, args.folder, args.name, poisson, plot)
    elif args.mode == "rigid3d":
        np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img = rec.get_kps_decs(rgb_images)
        rec.rigid3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump,
                         args.folder, args.name, poisson, plot)
    else:
        np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img = rec.get_kps_decs(rgb_images)
        rec.trans3d_proc(point_clouds, rgb_images, depth_images, dump, args.folder, args.name, poisson, plot)
