import copy

import numpy as np
import open3d as o3d


def visualize_transformation(pcd1, pcd2, transformation):
    """
    updated version draw_registration_result(source, target, transformation) from
    http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html
    Args:
        pcd1:
        pcd2:
        transformation:

    Returns:

    """
    pcd1_temp = copy.deepcopy(pcd1)
    pcd2_temp = copy.deepcopy(pcd2)
    pcd1_temp.transform(transformation)
    pcd1_temp.paint_uniform_color([1, 0.706, 0])
    pcd2_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp])


def preprocess_point_cloud(pcd, voxel_size, radius_normal, max_nn_normal, radius_feature, max_nn_feature, plot=True):
    """

    Args:
        pcd:
        voxel_size:
        radius_normal:
        radius_feature:

    Returns:

    """
    pcd_copy = pcd.voxel_down_sample(voxel_size=voxel_size)  # copy.deepcopy(pcd)
    # pcd_copy.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_copy.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
    print(pcd_copy.dimension)
    if plot:
        o3d.visualization.draw_geometries([pcd_copy], point_show_normal=True)

    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_copy,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature))
    return pcd_copy, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    rad = np.radians(5)
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(True), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result
