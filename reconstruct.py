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
import homography_utils.q8 as q8
import homography_utils.q9 as q9
import homo3d

def get_kps_decs(rgb_images):
    sift = cv2.xfeatures2d.SIFT_create()
    np_kps_pre_img = []
    cv_kps_pre_img = []
    cv_des_pre_img = []
    for img in  rgb_images:
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img, None)

        orginal_points = np.array([(kp[idx].pt[0], kp[idx].pt[1]) for idx in range(len(kp))], dtype=int)

        np_kps_pre_img.append(orginal_points)
        cv_kps_pre_img.append(kp)
        cv_des_pre_img.append(des)

    return np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img

def make_3d_kps(point_clouds, np_kps_pre_img):
    """

    Args:
        point_clouds:
        np_kps_pre_img:

    Returns:

    """
    return [utils.get_3d_kps(point_clouds[i], np_kps_pre_img[i]) for i in range(len(np_kps_pre_img))]

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

def fpfh(point_clouds, dump=False, dump_folder=None, image_set_name=None):

    pcds = make_pcds(point_clouds, dump, dump_folder, image_set_name)
    all_results = []
    for i in range(1, len(pcds)):

        src_pcd, src_fpfh = o3d_utils.preprocess_point_cloud(pcds[i-1], 10, 30, 10, 100)
        tar_pcd, tar_fpfh = o3d_utils.preprocess_point_cloud(pcds[i], 10, 30, 10, 100)

        results = o3d_utils.execute_fast_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 10)
        # results = o3d_utils.execute_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 1)
        o3d_utils.visualize_transformation(src_pcd, tar_pcd, results.transformation)
        print(results)
        print(results.transformation)
        all_results.append(results)
        if dump:
            o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, i-1), pcds[i-1].transform(results.transformation))
            o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, i), pcds[i])


def display_alpha_mesh(pcd):
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def display_voxleiation(pcd):
    print('voxelization')
    N=20000
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])

def display_ball_point(pcd):
    pcd1_temp = pcd #pcd.voxel_down_sample(voxel_size=0.5)
    pcd1_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    radii = [150, 150, 150, 150]

    distances = pcd1_temp.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd1_temp, o3d.utility.DoubleVector([radius, radius * 2]))
    rec_mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([rec_mesh])
    o3d.io.write_triangle_mesh("copy_of_knot.ply", rec_mesh)

def homo3d_proc(point_clouds, rgb_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump=False, dump_folder=None, image_set_name=None):
    pcds = make_pcds(point_clouds, dump, dump_folder, image_set_name)

    kps_3d = make_3d_kps(point_clouds, np_kps_pre_img)

    all_results = []
    for i in range(1, len(pcds)):
        img1, kp1, des1 = rgb_images[i], cv_kps_pre_img[i], cv_des_pre_img[i]
        img2, kp2, des2 = rgb_images[i-1], cv_kps_pre_img[i-1], cv_des_pre_img[i-1]

        bf_matches = q8.mathching_skimage(img1, kp1, des1, img2, kp2, des2)
        H_matrix, matchs = q9.ransac_loop(img1, img2, kp1, kp2, bf_matches)

        m_kps1_3d = []
        m_kps2_3d = []

        for m in bf_matches:
            m_kps1_3d.append(kps_3d[i][m[0]])
            m_kps2_3d.append(kps_3d[i-1][m[1]])

        Hmatrix = homo3d.homo_rigid_transform_3D(np.array(m_kps1_3d), np.array(m_kps2_3d))

        o3d_utils.visualize_transformation(pcds[i], pcds[i-1], Hmatrix)
        print(Hmatrix)
        all_results.append(Hmatrix)
        if dump:
            o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, i), pcds[i].transform(Hmatrix))
            o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, i-1), pcds[i-1])

        display_ball_point(pcds[i]+pcds[i-1])
        #display_voxleiation(pcds[i]+pcds[i-1])
        #display_alpha_mesh(pcds[i]+pcds[i-1])

def depth_images_to_3d_pts(depth_images):
    return [utils.depth_to_voxel(img) for img in depth_images]

def depth_images_to_3d_pts_v2(depth_images):
    return [utils.posFromDepth(img) for img in depth_images]


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='./models/nyu.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='./image_sets/cars/*.jpg', type=str, help='Input filename or folder.')
    args = parser.parse_args()

    rgb_images, depth_images = dd.get_depth(args.model, args.input)

    np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img = get_kps_decs(rgb_images)
    point_clouds = depth_images_to_3d_pts(depth_images)
    #fpfh(point_clouds, True, "./image_sets/kitchen", "kitchen")
    homo3d_proc(point_clouds, rgb_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, True, "./image_sets/cars", "cars")

