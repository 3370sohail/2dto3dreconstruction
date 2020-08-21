import os

import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    #pixels = pixels[pixels[:, 2] != 0]  # filter out missing data

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

MIN_MATCH_COUNT = 10

if __name__ == "__main__":
    # depth_folder = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\00003\depth"
    # imgs = read_depth_folder(depth_folder)

    img1 = cv2.imread('car1.jpg')          # queryImage
    img2 = cv2.imread('car2.jpg') # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    # cv2.drawMatchesKnn expects list of lists as matches.
    # print(len(good), good[0], kp1[0].pt, des1[0])
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()
    

    img1_depth = cv2.imread('car1-depth.png', cv2.IMREAD_GRAYSCALE)
    img2_depth = cv2.imread('car2-depth.png', cv2.IMREAD_GRAYSCALE)
    img1_depth = depth_to_voxel(img1_depth).astype(np.float32)
    img2_depth = depth_to_voxel(img2_depth).astype(np.float32)
    print(len(img1_depth), len(img2_depth))
    retval, out, inliers = cv2.estimateAffine3D(img1_depth, img2_depth)
    print(out)

    # img1 = depth_to_voxel(img1, 50)
    # # voxel_to_csv(img1, r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\00003\points1.csv")
    #
    # img2 = cv2.imread(
    #     r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\00003\depth\0000050-000001635093.png",
    #     cv2.IMREAD_GRAYSCALE)
    # img2 = depth_to_voxel(img2, 50)
    # # voxel_to_csv(img2, r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\00003\points2.csv")
    #
    # img3 = cv2.imread(
    #     r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\00003\depth\0000090-000002969863.png",
    #     cv2.IMREAD_GRAYSCALE)
    # img3 = depth_to_voxel(img3, 50)
    # voxel_to_csv(img3, r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\00003\points3.csv")


