import os
import glob
import argparse
import matplotlib
import csv
import cv2
import numpy as np
import open3d as o3d


def affine(ins, out):
    # calculations
    l = len(ins)
    B = np.vstack([np.transpose(ins), np.ones(l)])
    D = 1.0 / np.linalg.det(B)
    entry = lambda r, d: np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))
    M = [[(-1) ** i * D * entry(R, i) for i in range(l)] for R in np.transpose(out)]
    A, t = np.hsplit(np.array(M), [l - 1])
    t = np.transpose(t)[0]
    # output
    print("Affine transformation matrix:\n", A)
    print("Affine transformation translation vector:\n", t)
    # unittests
    print("TESTING:")
    for p, P in zip(np.array(ins), np.array(out)):
        image_p = np.dot(A, p) + t
        result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
        print(p, " mapped to: ", image_p, " ; expected: ", P, result)

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

def get_transformed_pointsv2(keypoints, H_matrix):
    keypoints_3dim = np.zeros((keypoints.shape[0], keypoints.shape[1] + 1))
    transformed_points = np.zeros(keypoints.shape)
    # print(keypoints_3dim.shape)
    for i in range(len(keypoints)):
        keypoint_3dim = np.array([[keypoints[i][1], keypoints[i][0], keypoints[i][2]]], dtype=float)
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


'''
def get_transformed_points(keypoints, H_matrix):
    transformed_points = np.zeros(keypoints.shape)
    #print(keypoints_3dim.shape)
    for i in range(len(keypoints)):
        keypoint_3dim = np.array([keypoints[i]], dtype=float)
        #print(np.transpose(keypoint_3dim))
        transformed_point_3dim = np.dot(H_matrix, np.transpose(keypoint_3dim))
        transformed_points[i] = np.transpose(transformed_point_3dim)

    return transformed_points
'''


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
from layers import BilinearUpSampling2D
from utils.utils import predict, load_images
import q9, q8, fpfh, r3d
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='bottle/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images(glob.glob(args.input))
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)
print(len(outputs), type(outputs))

files = glob.glob(args.input)

# matplotlib problem on ubuntu terminal fix
matplotlib.use('TkAgg')

cv2_imgs = []
for f in files:
    cv2_imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))

print(cv2_imgs[0])

img1 = cv2.resize(cv2_imgs[0], (320, 240))
img2 = cv2.resize(cv2_imgs[1], (320, 240))

# Display results
# fig, ax = plt.subplots(nrows=5, ncols=4)
c = 0
input_voxels = []
for i in range(len(inputs)):
    plt.imshow(outputs[i])
    plt.show()
    print(outputs[i].shape)
    voxel = depth_to_voxel(outputs[i])
    input_voxels.append(voxel)


# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

orginal_points = np.array([(kp1[idx].pt[1], kp1[idx].pt[0]) for idx in range(len(kp1))], dtype=int)
keypoints_prime = np.array([(kp2[idx].pt[1], kp2[idx].pt[0]) for idx in range(len(kp2))], dtype=int)
kps = [orginal_points, keypoints_prime]
print("des size", len(des1), len(des2))

bf_matches = q8.mathching_skimage(img1, kp1, des1, img2, kp2, des2, True, 135, 135, 50)
H_matrix, matchs = q9.ransac_loop(img1, img2, kp1, kp2, bf_matches)
matchs = bf_matches


m_kp1 = []
m_kp2 = []

for m in matchs:
    m_kp1.append(orginal_points[m[0]])
    m_kp2.append(keypoints_prime[m[1]])

kps1_3d = get_3d_kps(input_voxels[0], orginal_points)
kps2_3d = get_3d_kps(input_voxels[1], keypoints_prime)

m_kps1_3d = []
m_kps2_3d = []

for m in bf_matches:
    m_kps1_3d.append(kps1_3d[m[0]])
    m_kps2_3d.append(kps2_3d[m[1]])


R, t = r3d.rigid_transform_3D( np.transpose(get_3d_kps(input_voxels[1], m_kp2)), np.transpose(get_3d_kps(input_voxels[0], m_kp1)))
#affine(get_3d_kps(input_voxels[0], m_kp1), get_3d_kps(input_voxels[1], m_kp2))
print(R, t)
Hmatrix = np.concatenate((R, t), axis=1)
print(Hmatrix)
TR = R @ t
new_t = np.transpose(t)
new_3d_pts = [input_voxels[0]]
new_3d_pts.append(get_transformed_pointsv2(input_voxels[1], R))
#new_3d_pts.append(get_transformed_points(input_voxels[1], Hmatrix))
pcds = []
for i in range(len(inputs)):
    voxel_to_csv( input_voxels[i], './cars2/depth/car_{}.csv'.format(i))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_3d_pts[i])
    pcds.append(pcd)
    o3d.io.write_point_cloud("./cars2/depth/car_{}.pcd".format(i), pcd)

fpfh.draw_registration_result(pcds[0], pcds[1], np.identity(4))