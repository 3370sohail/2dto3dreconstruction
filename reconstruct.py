import argparse

import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import homography_utils.q8 as q8
import homography_utils.q9 as q9
import open3d_utils.fpfh as o3d_utils
import utils.r3d as r3d
import utils.transformation3d as trans3d
# import dense_depth.depth as dd
import utils.utils as utils


def get_kps_decs(rgb_images):
    """
    Generate SIFT descriptors for lists of RGB images

    :param rgb_images: list of (l, w, 3) images
    :return: list of keypoints in ndarray format, list of keypoints in OpenCV format, list of SIFT descriptors
    """
    sift = cv2.xfeatures2d.SIFT_create()
    np_kps_pre_img = []
    cv_kps_pre_img = []
    cv_des_pre_img = []

    # find the keypoints and descriptors with SIFT for every image
    for img in rgb_images:
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


def make_pcds(point_clouds, save_intermediate=False, out_folder=None, image_set_name=None):
    """
    Create Open3D PCD objects from list of point clouds

    :param point_clouds: list of (n, 3) ndarrays, where the 3 represents the x y z of each point
    :param save_intermediate: True to save the intermediate point clouds as lists of points before converting to PCD
    :param out_folder: folder to save point clouds (CSVs) to
    :param image_set_name: name root for point clouds (CSVs)
    :return: generated PCDs
    """
    pcds = []

    for i in range(len(point_clouds)):
        if save_intermediate:
            utils.voxel_to_csv(point_clouds[i], '{}/{}_{}.csv'.format(out_folder, image_set_name, i))

        # convert from ndarray to PCD
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds[i])
        pcds.append(pcd)

    return pcds


def fpfh(point_clouds, save_intermediate=False, out_folder=None, image_set_name=None, poisson=True, plot=True):
    """
    Global point cloud registration using Fast Point Feature Histogram (FPFH), using Open3D's

        - Implementation for FPFH descriptors: http://www.open3d.org/docs/0.9.0/python_api/open3d.registration.compute_fpfh_feature.html#open3d-registration-compute-fpfh-feature
        - RANSAC 3D transformation estimation: http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#RANSAC
        - Fast Global Registration: http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#Fast-global-registration

    We used these in order to make sure that the algorithm is both correct and optimized, and to get the best
    results from these complex algorithms.

    :param point_clouds: list of (l x w, 3) point clouds
    :param save_intermediate: True to save the intermediate point clouds when registering many point clouds
    :param out_folder: folder to save point clouds (CSVs and PCDs) and meshes (PLYs) to
    :param image_set_name: name root for point clouds (CSVs and PCDs) and meshes (PLYs)
    :param poisson: True to use Poisson surface reconstruction, False to use ball point surface reconstruction when
                    building the mesh
    :param plot: True to plot intermediate results when running algorithm, False otherwise
    :return: None, images will be saved to the out_folder
    """
    pcds = make_pcds(point_clouds, save_intermediate, out_folder, image_set_name)
    all_results = []

    # perform FPFH between every 2 consecutive images
    for i in range(1, len(pcds)):

        # compute PCDs and FPFH descriptors
        src_pcd, src_fpfh = o3d_utils.preprocess_point_cloud(pcds[i], 5, 10, 30, 25, 100, plot)
        tar_pcd, tar_fpfh = o3d_utils.preprocess_point_cloud(pcds[i - 1], 5, 10, 30, 25, 100, plot)

        # perform global registration between computed point clouds with features
        results = o3d_utils.execute_global_registration(src_pcd, tar_pcd, src_fpfh, tar_fpfh, 5)

        if plot:
            o3d_utils.visualize_transformation(src_pcd, tar_pcd, results.transformation)

        print(results)
        print(results.transformation)
        all_results.append(results.transformation)

    # chain all point clouds together with computed transformation
    chain_transformation(pcds, all_results, save_intermediate, out_folder, image_set_name, poisson, plot)


def apply_ball_point(pcd, plot=True):
    """
    Apply ball point surface reconstruction using Open3D library. Full documentation and implementation here:

    http://www.open3d.org/docs/release/tutorial/Advanced/surface_reconstruction.html#Ball-pivoting

    :param pcd: point cloud object
    :param plot: True to plot intermediate results when running algorithm, False otherwise
    :return: generated mesh from point cloud
    """
    print("applying ball point surface reconstruction")
    pcd1_temp = pcd  # pcd.voxel_down_sample(voxel_size=0.5)
    pcd1_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    radii = [150, 150, 150, 150]

    # find radii size around points
    distances = pcd1_temp.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    # generate ball point surface meshing
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd1_temp, o3d.utility.DoubleVector([radius, radius * 2])
    )

    # colour and return
    rec_mesh.paint_uniform_color([1, 0.706, 0])
    if plot:
        o3d.visualization.draw_geometries([rec_mesh])
    return rec_mesh


def apply_poisson(pcd, plot=True):
    """
    Apply Poisson surface reconstruction using Open3D library. Full documentation and implementation here:

    http://www.open3d.org/docs/release/tutorial/Advanced/surface_reconstruction.html#Poisson-surface-reconstruction

    :param pcd: point cloud object
    :param plot: True to plot intermediate results when running algorithm, False otherwise
    :return: generated mesh from point cloud
    """
    # generate Poisson surface meshing
    print("applying poisson surface reconstruction")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # filter out areas that are too dense
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # colour and return
    mesh.paint_uniform_color([1, 0.706, 0])
    if plot:
        o3d.visualization.draw_geometries([mesh])
    return mesh


def chain_transformation(pcds, transformations, save_intermediate=False, out_folder=None, image_set_name=None, poisson=True,
                         plot=True):
    """
    Chain together point clouds from list of transformations, generate a mesh, and save

    :param pcds: list of point clouds to merge
    :param transformations: list of (len(pcds) - 1) 4x4 transformation matrices, each matrix at index i should
                            transform point cloud pcds[i] onto point cloud pcds[i + 1]
    :param save_intermediate: True to save the intermediate point clouds when registering many point clouds
    :param out_folder: folder to save point clouds (CSVs and PCDs) and meshes (PLYs) to
    :param image_set_name: name root for point clouds (CSVs and PCDs) and meshes (PLYs)
    :param poisson: True to use Poisson surface reconstruction, False to use ball point surface reconstruction when
                    building the mesh
    :param plot: True to plot intermediate results when running algorithm, False otherwise
    :return: None, images will be saved to the out_folder
    """
    combined_pcd = pcds[0]

    if save_intermediate:
        o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(out_folder, image_set_name, 0), pcds[0])

    # merge point clouds using transformation matrices
    for i in range(1, len(pcds)):
        for j in range(i, 0, -1):
            pcds[i].transform(transformations[j - 1])

        if save_intermediate:
            o3d.io.write_point_cloud('{}/{}_{}.pcd'.format(out_folder, image_set_name, i), pcds[i])

        combined_pcd += pcds[i]

    # generate surface mesh for merged point clouds
    combined_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if poisson:
        mesh = apply_poisson(combined_pcd, plot)
        name = 'poisson'
    else:
        mesh = apply_ball_point(combined_pcd, plot)
        name = 'ball_point'

    # save mesh
    o3d.io.write_triangle_mesh('{}/{}_{}_{}_mesh.ply'.format(out_folder, image_set_name, "final", name), mesh)
    print('saved final to {}/{}_{}_{}_mesh.ply'.format(out_folder, image_set_name, "final", name))


def rigid3d_proc(point_clouds, rgb_images, depth_images, np_kps_pre_img, cv_kps_pre_img, cv_des_pre_img,
                 save_intermediate=False, out_folder=None, image_set_name=None, poisson=True, plot=True):
    """
    Global point cloud registration by computing the 3D transformation matrix between pairs of point clouds and
    then further refining with ICP. See trans3d.register_imgs() for full documentation on how this process works.

    :param point_clouds: list of (l x w, 3) point clouds
    :param rgb_images: list of (l, w, 3) RGB images
    :param depth_images: list of (l, w, 1) depth images
    :param np_kps_pre_img: list of keypoints in ndarray format
    :param cv_kps_pre_img: list of keypoints in OpenCV format
    :param cv_des_pre_img: list of SIFT descriptors
    :param save_intermediate: True to save the intermediate point clouds when registering many point clouds
    :param out_folder: folder to save point clouds (CSVs and PCDs) and meshes (PLYs) to
    :param image_set_name: name root for point clouds (CSVs and PCDs) and meshes (PLYs)
    :param poisson: True to use Poisson surface reconstruction, False to use ball point surface reconstruction when
                    building the mesh
    :param plot: True to plot intermediate results when running algorithm, False otherwise
    :return: None, images will be saved to the out_folder
        """
    pcds = make_pcds(point_clouds, save_intermediate, out_folder, image_set_name)
    kps_3d = make_3d_kps_depth_img(depth_images, np_kps_pre_img)
    all_results = []

    # perform global registration between every 2 consecutive images
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

    # chain all point clouds together with computed transformation
    chain_transformation(pcds, all_results, save_intermediate, out_folder, image_set_name, poisson, plot)


def trans3d_proc(point_clouds, rgb_images, depth_images, save_intermediate=False, out_folder=None, image_set_name=None,
                 poisson=True, plot=True, filter_pts_frac=0.1, partial_set_frac=0.7):
    """
    Global point cloud registration by computing the 3D transformation matrix between pairs of point clouds and
    then further refining with ICP. See trans3d.register_imgs() for full documentation on how this process works.

    :param point_clouds: list of (l x w, 3) point clouds
    :param rgb_images: list of (l, w, 3) RGB images
    :param depth_images: list of (l, w, 1) depth images
    :param save_intermediate: True to save the intermediate point clouds when registering many point clouds
    :param out_folder: folder to save point clouds (CSVs and PCDs) and meshes (PLYs) to
    :param image_set_name: name root for point clouds (CSVs and PCDs) and meshes (PLYs)
    :param poisson: True to use Poisson surface reconstruction, False to use ball point surface reconstruction when
                    building the mesh
    :param plot: True to plot intermediate results when running algorithm, False otherwise
    :param filter_pts_frac: see trans3d.register_imgs() for documentation. The default value is what we found works
    :param partial_set_frac: see trans3d.register_imgs() for documentation. The default value is what we found works
    :return: None, images will be saved to the out_folder
    """
    pcds = make_pcds(point_clouds, save_intermediate, out_folder, image_set_name)
    all_results = []

    # perform global registration between every 2 consecutive images
    for i in range(1, len(pcds)):

        # global registration with 3D transformation matrix and local fine registration with ICP
        _, _, h = trans3d.register_imgs(rgb_images[i], rgb_images[i - 1], depth_images[i], depth_images[i - 1],
                                        img1_pts=point_clouds[i], img2_pts=point_clouds[i - 1],
                                        filter_pts_frac=filter_pts_frac, partial_set_frac=partial_set_frac)

        if plot:
            o3d_utils.visualize_transformation(pcds[i], pcds[i - 1], h)

        print(h)
        all_results.append(h)

    # chain all point clouds together with computed transformation
    chain_transformation(pcds, all_results, save_intermediate, out_folder, image_set_name, poisson, plot)


def depth_images_to_3d_pts(depth_images, scale=1.):
    """
    Convert list of depth images to 3D points

    :param depth_images: list of (l, w, 1) depth images
    :param scale: constant to scale depth down by
    :return: list of (l x w, 3) ndarrays
    """
    return [utils.depth_to_voxel(img, scale) for img in depth_images]


def depth_images_to_3d_pts_v2(depth_images):
    """
    Convert list of depth images to 3D points

    :param depth_images: list of (l, w, 1) depth images
    :return: list of (l x w, 3) ndarrays
    """
    return [utils.posFromDepth(img) for img in depth_images]


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='./models/nyu.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='./image_sets/kitchen2/*.png', type=str, help='Input filename or folder.')
    parser.add_argument('--mode', default='homo', type=str, help='method of reconstruction')
    parser.add_argument('--surface', default='poisson', type=str, help='method of reconstruction')
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
        trans3d_proc(point_clouds, rgb_images, depth_images, dump, args.folder, args.name, poisson, plot)
