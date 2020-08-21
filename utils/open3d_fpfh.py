import copy

import numpy as np
import open3d as o3d

"""
DISCLAIMER: this code belongs to the open3d tutorial
http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html
with only very small modifications for tuning proposes 
"""

def visualize_transformation(pcd1, pcd2, transformation):
    """
    credit to open3d tutorial:
        - http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#Visualization
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
    credit to open3d tutorial:
        - Implementation for FPFH descriptors: http://www.open3d.org/docs/0.9.0/python_api/open3d.registration.compute_fpfh_feature.html#open3d-registration-compute-fpfh-feature
    Args:
        pcd:
        voxel_size:
        radius_normal:
        radius_feature:

    Returns:

    """
    pcd_copy = pcd.voxel_down_sample(voxel_size=voxel_size)
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
    """
    credit to open3d tutorial:
        - RANSAC 3D transformation estimation: http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#RANSAC
    Args:
        source_down:
        target_down:
        source_fpfh:
        target_fpfh:
        voxel_size:

    Returns:

    """
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(True), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(5000000, 1000))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    """
    credit to open3d tutorial:
        - Fast Global Registration: http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#Fast-global-registration
    Args:
        source_down:
        target_down:
        source_fpfh:
        target_fpfh:
        voxel_size:

    Returns:

    """
    distance_threshold = voxel_size * 1.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result
