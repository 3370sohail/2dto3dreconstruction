import os
import glob
import matplotlib
import csv
import open3d as o3d
from skimage.measure import ransac


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
    #print(keypoints_3dim.shape)
    for i in range(len(keypoints)):
        keypoint_3dim = np.array([[keypoints[i][1], keypoints[i][0], keypoints[i][2], 1]], dtype=float)
        #print(np.transpose(keypoint_3dim))
        transformed_point_3dim = np.dot(H_matrix, np.transpose(keypoint_3dim))
        #print(transformed_point_3dim)
        
        #normalize_transformed_point = transformed_point_3dim[:4, :] / transformed_point_3dim[3, :]
        #keypoints_3dim[i] = transformed_point_3dim
        new_value = np.transpose(transformed_point_3dim)
        transformed_points[i] = [new_value[0][1], new_value[0][0], new_value[0][2]]
        #print(transformed_points)

    #print(keypoints_3dim)
    #print(transformed_points)
    #print(keypoints)
    return transformed_points

def find_closest_3d_match(x0, y0, matrix_3d):
    arry_x0_y0 = [(x0, y0) for i in range(len(matrix_3d))]
    index = np.argmin(np.sum(((matrix_3d[:, :2] - arry_x0_y0) ** 2), axis=0))
    return matrix_3d[index]

def get_3d_kps(voxels, kps):
    kps_3d = []
    for i in kps:
        for v in voxels:
            if(i[1] == v[0] and i[0] == v[1]):
                kps_3d.append(v)
                break

    return np.array(kps_3d)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2




    

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
import q8, q9, structure
from matplotlib import pyplot as plt


import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('cars/car1.jpg',0)  #queryimage # left image
img2 = cv2.imread('cars/car2.jpg',0) #trainimage # right image

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

orginal_points = np.array([(kp1[idx].pt[1], kp1[idx].pt[0], 1) for idx in range(len(kp1))], dtype=int).T
keypoints_prime = np.array([(kp2[idx].pt[1], kp2[idx].pt[0], 1) for idx in range(len(kp2))], dtype=int).T

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
points1 = np.array([(pts1[idx][1], pts1[idx][0], 1) for idx in range(len(pts1))], dtype=int).T
points2 = np.array([(pts2[idx][1], pts2[idx][0], 1) for idx in range(len(pts2))], dtype=int).T



P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = structure.compute_P_from_fundamental(F)

#points1n = np.dot(np.linalg.inv(F), points1)
#points2n = np.dot(np.linalg.inv(F), points2)

ind = -1
# for i, P2 in enumerate(P2s):
#     # Find the correct camera parameters
#     print("PARAM1", points1n[:, 0], "PARAM2", points2n[:, 0], "PARAM3", P1, "PARAM4", P2)
#     d1 = structure.reconstruct_one_point(
#         points1n[:, 0], points2n[:, 0], P1, P2)

#     # Convert P2 from camera view to world view
#     P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
#     d2 = np.dot(P2_homogenous[:3, :4], d1)

#     if d1[2] > 0 and d2[2] > 0:
#         ind = i


# P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]

#tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
print(points1.shape)
print(points2.shape)

tripoints3d = structure.linear_triangulation(points1, points2, P1, P2)

fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')
ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()


# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()