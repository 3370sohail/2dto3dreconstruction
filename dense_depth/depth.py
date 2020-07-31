import os
import glob
import argparse
import csv
import cv2
import numpy as np


def read_depth_folder(path):
    """
    Read all the images from a folder

    :param path: path to folder to read
    :return: list of loaded grayscale images
    """
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    imgs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in files]
    return imgs


def depth_to_voxel(img, scale=1):
    """
    Given a depth image, convert all the points in the image to 3D points

    NOTE ON SCALE:
        The values in 3D space are not necessarily to scale. For example a car might be a meter away in
        real life, but on the depth map it only has a value of 10. We therefore need to give it a scale
        value to multiply this depth by to get its actual depth in 3D space. This scale value can be
        estimated by looking at how long or wide the actual object should be, and then scaling accordingly.

    :param img: ndarray representing depth values in image
    :param scale: how far away every value is--a number to multiply the depth values by
    :return: n x 3 ndarray, where n is the number of 3D points, and each of the 3 represents the value
             in that dimension
    """
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to n x 3
    pixels = np.stack((xx, yy, img.astype(np.int16) * scale), axis=2)
    pixels = np.reshape(pixels, (img.shape[0] * img.shape[1], 3))
    pixels = pixels[pixels[:, 2] != 0]  # filter out missing data

    return pixels


def voxel_to_csv(points, path):
    """
    Write points to csv file

    :param points: n x 3 ndarray
    :param path: path to csv file to save to
    :return: None
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(points)


def get_transformed_points(keypoints, H_matrix):
    keypoints_3dim = np.zeros((keypoints.shape[0], keypoints.shape[1] + 1))
    transformed_points = np.zeros(keypoints.shape)
    # print(keypoints_3dim.shape)
    for i in range(len(keypoints)):
        keypoint_3dim = np.array([[keypoints[i][1], keypoints[i][0], keypoints[i][2], 1]], dtype=float)
        # print(np.transpose(keypoint_3dim))
        transformed_point_3dim = np.dot(H_matrix, np.transpose(keypoint_3dim))
        # print(transformed_point_3dim)

        # normalize_transformed_point = transformed_point_3dim[:4, :] / transformed_point_3dim[3, :]
        # keypoints_3dim[i] = transformed_point_3dim
        new_value = np.transpose(transformed_point_3dim)
        transformed_points[i] = [new_value[0][1], new_value[0][0], new_value[0][2]]
        # print(transformed_points)

    # print(keypoints_3dim)
    # print(transformed_points)
    # print(keypoints)
    return transformed_points


def find_closest_3d_match(x0, y0, matrix_3d):
    arry_x0_y0 = [(x0, y0) for i in range(len(matrix_3d))]
    index = np.argmin(np.sum(((matrix_3d[:, :2] - arry_x0_y0) ** 2), axis=0))
    return matrix_3d[index]


def get_3d_kps(voxels, kps):
    kps_3d = []
    for i in kps:
        for v in voxels:
            if (i[1] == v[0] and i[0] == v[1]):
                kps_3d.append(v)
                break

    return np.array(kps_3d)


# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from dense_depth.layers import BilinearUpSampling2D
from dense_depth.utils import predict, load_images, display_images

from matplotlib import pyplot as plt

def get_depth(model, image_set, plot=False):
    """
    Note This code is used to contact the depth detection model from
    https://github.com/ialhashim/DenseDepth
    Args:
        model:
        image_set:
        plot:

    Returns: List of rgb images, list of depth images

    """
    print('Loading model...')

    print('\nModel loaded ({0}).'.format(model))

    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    # Load model into GPU / CPU
    model = load_model(model, custom_objects=custom_objects, compile=False)


    # Input images
    inputs = load_images(glob.glob(image_set))
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)
    print(len(outputs), type(outputs))

    files = glob.glob(image_set)

    cv2_imgs = []
    for f in files:
        cv2_imgs.append(cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), (320, 240)))


    # rgb and depth images

    if plot:
        for i in range(len(files)):
            fig, axs = plt.subplots(2)
            axs[0].imshow(cv2_imgs[i], cmap="gray")
            axs[1].imshow(outputs[i])

            plt.show()

    return cv2_imgs, outputs

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--model', default='kitti.h5', type=str, help='Trained Keras model file.')
    parser.add_argument('--input', default='../image_sets/helmet/*.jpg', type=str, help='Input filename or folder.')
    args = parser.parse_args()

    get_depth(args.model, args.input, plot=True)