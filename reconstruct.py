import numpy as np
import cv2
from matplotlib import pyplot as plt
import open3d as o3d

import argparse

import dense_depth.depth as dd
import utils.utils as utils
import open3d_utils.fpfh as o3d_utils
import homography_utils.q8 as q8
import homography_utils.q9 as q9
import homo3d
import utils.r3d as r3d


def get_kps_decs(rgb_images):
    sift = cv2.xfeatures2d.SIFT_create()
    np_kps_pre_img = []
    cv_kps_pre_img = []
    cv_des_pre_img = []
    for img in rgb_images:
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


def make_3d_kps_depth_img(depth_images, np_kps_pre_img):
    kps_3d = []

    for i in range(len(depth_images)):
        img_kp = [(p[0], p[1], depth_images[i][p[1], p[0]]) for p in np_kps_pre_img[i]]
        kps_3d.append(img_kp)

    return kps_3d


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
        #if dump:
        #    utils.voxel_to_csv(point_clouds[i], '{}/{}_{}.csv'.format(dump_folder, image_set_name, i))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds[i])
        pcds.append(pcd)

    return pcds


def fpfh(point_clouds, dump=False, dump_folder=None, image_set_name=None, poisson=True, plot=True):
    pcds = make_pcds(point_clouds, dump, dump_folder, image_set_name)
    all_results = []
    for i in range(1, len(pcds)):

        src_pcd, src_fpfh = o3d_utils.preprocess_point_cloud(pcds[i], 5, 10, 30, 25, 100, plot)
        tar_pcd, tar_fpfh = o3d_utils.preprocess_point_cloud(pcds[i - 1], 5, 10, 30, 25, 100, plot)

        # results = o3d_utils.execute_fast_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 5)
        results = o3d_utils.execute_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 5)
        if plot:
            o3d_utils.visualize_transformation(src_pcd, tar_pcd, results.transformation)
        print(results)
        print(results.transformation)
        all_results.append(results.transformation)

    chain_transformation(pcds, all_results, dump, dump_folder, image_set_name, poisson, plot)


def display_alpha_mesh(pcd):
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def display_voxleiation(pcd, plot):
    print('voxelization')
    #N = 20000
    #pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=1)
    if plot:
        o3d.visualization.draw_geometries([voxel_grid])



def apply_ball_point(pcd, plot=True):
    print("applying ball point surface reconstruction")
    pcd1_temp = pcd
    pcd1_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # idea for avg distance from https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices
    distances = pcd1_temp.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd1_temp, o3d.utility.DoubleVector(
        [radius, radius * 1.5, radius * 2]))
    rec_mesh.paint_uniform_color([1, 0.706, 0])
    if plot:
        o3d.visualization.draw_geometries([rec_mesh])
    return rec_mesh


def apply_poisson(pcd, plot=True):
    print("applying poisson surface reconstruction")
    pcd = pcd.voxel_down_sample(voxel_size=1)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.paint_uniform_color([1, 0.706, 0])
    if plot:
        o3d.visualization.draw_geometries([mesh])
    return mesh


def chain_transformation(pcds, transformations, dump=False, dump_folder=None, image_set_name=None, poisson=True,
                         plot=True):
    """

    Args:
        pcds:
        transformations:
        dump:
        dump_folder:
        image_set_name:
        pisson:

    Returns:

    """

    combined_pcd = pcds[0]
    if dump:
        o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, 0), pcds[0])
    for i in range(1, len(pcds)):

        for j in range(i, 0, -1):
            pcds[i].transform(transformations[j - 1])

        if dump:
            o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, i), pcds[i])

        combined_pcd += pcds[i]
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=2)
        print("donwsmapling ", i, print(combined_pcd.dimension))

    o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(dump_folder, image_set_name, "final"), combined_pcd)
    combined_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if poisson:
        mesh = apply_poisson(combined_pcd, plot)
        name = 'poisson'
    else:
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=10)
        mesh = apply_ball_point(combined_pcd, plot)
        name = 'ball_point'

    o3d.io.write_triangle_mesh('{}/{}_{}_{}_mesh.ply'.format(dump_folder, image_set_name, "final", name), mesh)
    print('saved final to {}/{}_{}_{}_mesh.ply'.format(dump_folder, image_set_name, "final", name))


def rigid3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump=False,
                 dump_folder=None, image_set_name=None, poisson=True, plot=True):
    pcds = make_pcds(point_clouds, dump, dump_folder, image_set_name)

    kps_3d = make_3d_kps_depth_img(depth_images, np_kps_pre_img)

    all_results = []
    for i in range(1, len(pcds)):
        img1, kp1, des1 = rgb_images[i], cv_kps_pre_img[i], cv_des_pre_img[i]
        img2, kp2, des2 = rgb_images[i - 1], cv_kps_pre_img[i - 1], cv_des_pre_img[i - 1]

        bf_matches = q8.mathching_skimage(img1, kp1, des1, img2, kp2, des2, plot)
        H_matrix, matchs = q9.ransac_loop(img1, img2, kp1, kp2, bf_matches)

        m_kps1_3d = []
        m_kps2_3d = []

        for m in matchs:
            m_kps1_3d.append(kps_3d[i][m[0]])
            m_kps2_3d.append(kps_3d[i - 1][m[1]])

        R, t = r3d.rigid_transform_3D(np.array(m_kps1_3d).T, np.array(m_kps2_3d).T)
        Hmatrix = np.pad(R, ((0, 1), (0, 1)))
        Hmatrix[3, 3] = 1
        Hmatrix[0, 3] = t[0, 0]
        Hmatrix[1, 3] = t[1, 0]
        Hmatrix[2, 3] = t[2, 0]
        print(t)
        if plot:
            o3d_utils.visualize_transformation(pcds[i], pcds[i - 1], Hmatrix)
        print(Hmatrix)
        all_results.append(Hmatrix)

    chain_transformation(pcds, all_results, dump, dump_folder, image_set_name, poisson, plot)
    display_voxleiation(pcds[0])


def homo3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump=False,
                dump_folder=None, image_set_name=None, poisson=True, plot=True):
    pcds = make_pcds(point_clouds, dump, dump_folder, image_set_name)

    kps_3d = make_3d_kps_depth_img(depth_images, np_kps_pre_img)

    all_results = []
    for i in range(1, len(pcds)):
        img1, kp1, des1 = rgb_images[i], cv_kps_pre_img[i], cv_des_pre_img[i]
        img2, kp2, des2 = rgb_images[i - 1], cv_kps_pre_img[i - 1], cv_des_pre_img[i - 1]

        bf_matches = q8.mathching_skimage(img1, kp1, des1, img2, kp2, des2, plot)
        H_matrix, matchs = q9.ransac_loop(img1, img2, kp1, kp2, bf_matches)

        m_kps1_3d = []
        m_kps2_3d = []

        for m in matchs:
            m_kps1_3d.append(kps_3d[i][m[0]])
            m_kps2_3d.append(kps_3d[i - 1][m[1]])

        Hmatrix = homo3d.homo_rigid_transform_3D(np.array(m_kps1_3d), np.array(m_kps2_3d))

        if plot:
            o3d_utils.visualize_transformation(pcds[i], pcds[i - 1], Hmatrix)
        print(Hmatrix)
        all_results.append(Hmatrix)

    chain_transformation(pcds, all_results, dump, dump_folder, image_set_name, poisson, plot)


def depth_images_to_3d_pts(depth_images):
    return [utils.depth_to_voxel(img, 1) for img in depth_images]


def depth_images_to_3d_pts_v2(depth_images):
    return [utils.posFromDepth(img) for img in depth_images]


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='./models/nyu.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='./image_sets/cars/*.jpg', type=str, help='Input filename or folder.')
    parser.add_argument('--mode', default='fpfh', type=str, help='method of reconstruction')
    parser.add_argument('--surface', default='ball', type=str, help='method of reconstruction')
    parser.add_argument('--dump', default='yes', type=str, help='method of reconstruction')
    parser.add_argument('--folder', default='./image_sets/cars', type=str, help='method of reconstruction')
    parser.add_argument('--name', default='cars', type=str, help='method of reconstruction')
    parser.add_argument('--plot', default='no', type=str, help='method of reconstruction')

    args = parser.parse_args()

    rgb_images, depth_images = dd.get_depth(args.model, args.input)

    plot = False
    if args.plot == "yes":
        plot = True

    if plot:
        fig, axs = plt.subplots(len(rgb_images), 2)
        for i in range(len(rgb_images)):
            axs[i, 0].imshow(rgb_images[i])
            axs[i, 1].imshow(depth_images[i])

        plt.show()

    poisson = False
    if args.surface == "poisson":
        poisson = True

    dump = False
    if args.dump == "yes":
        dump = True

    np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img = get_kps_decs(rgb_images)
    point_clouds = depth_images_to_3d_pts(depth_images)
    if (args.mode == "fpfh"):
        fpfh(point_clouds, dump, args.folder, args.name, poisson, plot)
    elif (args.mode == "rigid3d"):
        rigid3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump,
                     args.folder, args.name, poisson, plot)
    else:
        homo3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img, dump,
                    args.folder, args.name, poisson, plot)
