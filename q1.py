import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu, sobel_h, sobel_v, gaussian
from skimage.feature import blob_dog, corner_harris, corner_peaks
import argparse
import time

from collections import namedtuple
Point = namedtuple('Point', 'x y sigma')
Siftpoint = namedtuple('Siftpoint', 'x y sigma n_gradient_m n_gradient_o n_gradient_g_m')
Siftvector = namedtuple('Siftvector', 'x y sigma w_array')


def create_2d_gaussian_matrix(sigma, size):
    """
    creates a 2d gaussian matrix
    :param sigma: sigma to use in the gaussian function
    :param size: size of width and height of the matrix
    :return: a numpy 2d gaussian matrix
    """

    matrix_size = size
    gaussian_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float)
    matrix_movement = matrix_size // 2 # since we only deal with evenly shaped matrixs

    for x in range(-matrix_movement, matrix_movement+1):
        for y in range(-matrix_movement, matrix_movement+1):
            gaussian_matrix[x + matrix_movement, y + matrix_movement] = (1 / (2 * math.pi * (sigma**2))) * math.exp(-(x**2 + y**2)/(2 * sigma**2))

    return gaussian_matrix

def image_pyramid(img, level):
    """
    using the given img create an image pyramid where each has a
    laplacian of gaussian applied to it based on the current level
    :param img: the img to make the pyramid based on
    :param level: number of level in the pyramid
    :return: array of images where each img is a level in the image pyramid
    """

    image_results = []
    
    image_results.append(gaussian(img, 0, preserve_range=True))

    for n in range(1, level):
        scale_factor = 2 ** n
        img_blurred = gaussian(img, scale_factor, preserve_range=True) 
        img_resized = resize(img_blurred, (int(img_blurred.shape[0] / scale_factor ), int(img_blurred.shape[1] / scale_factor)), anti_aliasing=False, preserve_range=True)
        image_results.append(img_resized)

    return image_results

def DoG(img_list_of_LoG):
    """
    compute the difference of gaussian of each image in the list by usign the level above it.
    :param img_list_of_LoG: list of gaussian blurred images
    :return: array of images where each img is difference of gaussian
    """
    image_results = []

    for n in range(len(img_list_of_LoG) -1):
        scale_factor = 2 ** n
        img1 = img_list_of_LoG[n]
        img2 = img_list_of_LoG[n+1]
        upsample_img2 = resize(img2, (int(img2.shape[0] * 2 ), int(img2.shape[1] * 2)), anti_aliasing=False, preserve_range=True)
        image_results.append(upsample_img2 - img1)
        print(image_results[n].sum())

    print("Dog size:", len(image_results))
    return image_results

def local_max_or_min(list_of_DoG, cur_level, neighborhood):
    """
    find all the local min and max value in both space and scale
    :param list_of_DoG: list of difference of gaussian images
    :param cur_level: current level in the list of difference of gaussian images we arr at
    :param neighborhood: size of the neighborhood around each local min and max value we will check
    :return: list of tuples of type Point
    """
    cur_level_img = list_of_DoG[cur_level]

    mean = np.mean(cur_level_img)
    std = np.std(cur_level_img)
    img_min = np.min(cur_level_img)
    img_max = np.max(cur_level_img)
    print(mean, std, img_min, img_max)

    intrest_points = []

    for i in range(neighborhood[0], cur_level_img.shape[0] - neighborhood[0]):
        for j in range(neighborhood[0], cur_level_img.shape[1] - neighborhood[0]):
            cur_value = cur_level_img[i,j]
            # find the max and min in the local window on the current level
            window = cur_level_img[i-(neighborhood[0]-1):i+neighborhood[0], j-(neighborhood[0]-1):j+neighborhood[0]]
            window_max_value = np.max(window)
            window_min_value = np.min(window)

            if(cur_value == window_max_value and cur_value > img_max * 0.5):

                scale_max_value = 0
                isScaleMax = True
                # check if the current value is extrema in scale space
                for k in range(max(0,cur_level-1),min(cur_level+1,len(list_of_DoG))):
                    cur_window = list_of_DoG[k][i-(neighborhood[0]-1):i+neighborhood[0], j-(neighborhood[0]-1):j+neighborhood[0]]
                    scale_max_value = np.max(cur_window)
                    if (scale_max_value > cur_value):
                        isScaleMax = False

                if (isScaleMax):
                    intrest_points.append(Point((i-neighborhood[0]), (j-neighborhood[0]), cur_level))

            if(cur_value == window_min_value and cur_value < img_min * 2):

                scale_min_value = 0
                isScaleMin = True
                # check if the current value is extrema in scale space
                for k in range(max(0,cur_level-1),min(cur_level+1,len(list_of_DoG))):
                    cur_window = list_of_DoG[k][i-(neighborhood[0]-1):i+neighborhood[0], j-(neighborhood[0]-1):j+neighborhood[0]]
                    scale_min_value = np.min(cur_window)
                    if (scale_min_value < cur_value):
                        isScaleMin = False

                if (isScaleMin):
                    intrest_points.append(Point((i-neighborhood[0]), (j-neighborhood[0]), cur_level))

    print(len(intrest_points))
    return intrest_points


def find_keypoints(list_of_DoG, neighborhood_size=5):
    """
    find key points in the list of difference of gaussian images
    :param list_of_DoG: list of difference of gaussian images
    :param neighborhood_size: size of the neighborhood around each local min and max value we will check
    :return: list of tuples of type Point, for each key point
    """

    intrest_points = []

    neighborhood = (neighborhood_size-1, neighborhood_size-1)

    list_of_DoGs_by_scale = {}

    pyramid_len = len(list_of_DoG) -1
    if (len(list_of_DoG) < 6):
        pyramid_len = pyramid_len + 1

    for i in range(pyramid_len):
        list_of_DoGs_by_scale[i] = []
        
        for j in range(pyramid_len):
            if (j < i):
                # note here I manually upsmaple as the built in one always seems to apply an algo to smooth to much etc
                img_cur_level = list_of_DoG[j][::2 ** (i-j) , ::2 ** (i-j)]
                pad_cur_img = np.pad(img_cur_level, (neighborhood, neighborhood))
            elif (j > i):
                # note here I manually subsmaple as the built in one always seems to apply an algo to smooth to much etc
                img_cur_level = list_of_DoG[j].repeat(2 ** (j-i), axis=0).repeat(2 ** (j-i), axis=1)
                pad_cur_img = np.pad(img_cur_level, (neighborhood, neighborhood))
            else:
                pad_cur_img = np.pad(list_of_DoG[j], (neighborhood, neighborhood))
            list_of_DoGs_by_scale[i].append(pad_cur_img)
        
        intrest_points += local_max_or_min(list_of_DoGs_by_scale[i], i, neighborhood)

    print(len(intrest_points))

    return intrest_points


def get_dx_dy_of_image_pyramid(image_pyramid):
    """
    compute the change in dx and dy for all the image in the image pyramid, using central differences
    :param image_pyramid: list of images
    :return: 2 arrays one for the dx and dy for all the image in the image pyramid
    """
    dx_of_image_pyramid = []
    dy_of_image_pyramid = []

    pyramid_len = len(image_pyramid) -1
    if (len(image_pyramid) < 6):
        pyramid_len = pyramid_len + 1


    for n in range(pyramid_len):
        # normalize each of the i
        out = cv2.normalize(image_pyramid[n].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) # image_pyramid[n].astype('float')
        # dx_of_image_pyramid.append(sobel_h(image_pyramid[n]))
        dx_of_image_pyramid.append(cv2.Sobel(out,cv2.CV_64F,1,0))
        # dy_of_image_pyramid.append(sobel_v(image_pyramid[n]))
        dy_of_image_pyramid.append(cv2.Sobel(out,cv2.CV_64F,0,1))

    return dx_of_image_pyramid, dy_of_image_pyramid

def get_gradient_data(keypoints, dx_of_image_pyramid, dy_of_image_pyramid):
    """
    for each keypoint using a 16x16 window compute 1) the gradient magnitude , 2) the gradient orientation as vectors,
    3) a 2D Gaussian weighted version of the gradient magnitude
    :param keypoints: list keypoints of type Point(), to process
    :param dx_of_image_pyramid: dx computed on the image_pyramid
    :param dy_of_image_pyramid: dy computed on the image_pyramid
    :return: list of SiftPoints
    """

    gradient_data = []

    gradient_mags = []
    gradient_ors = []
    for n in range(len(dy_of_image_pyramid)):
        mags = np.sqrt(np.square(dx_of_image_pyramid[n]) + np.square(dy_of_image_pyramid[n]))
        gradient_mags.append(np.pad(mags, ((9, 9), (9, 9)), 'constant'))
        orintation = np.arctan2(dy_of_image_pyramid[n], dx_of_image_pyramid[n]) #* (180/np.pi)
        gradient_ors.append(np.pad(orintation, ((9, 9), (9, 9)), 'constant'))

    gussian = create_2d_gaussian_matrix(4, 17)
    gussian_trim = gussian[0:-1,0:-1]

    for p in keypoints:
        n_gradient_mag = gradient_mags[p.sigma][p.x+9-8:p.x+9+8, p.y+9-8:p.y+9+8]
        n_gradient_or = gradient_ors[p.sigma][p.x+9-8:p.x+9+8, p.y+9-8:p.y+9+8]

        n_gradient_mag_w = n_gradient_mag * gussian_trim
        gradient_data.append(Siftpoint(p.x, p.y, p.sigma, n_gradient_mag, n_gradient_or, n_gradient_mag_w))

    return gradient_data

def reorgnize(list, index):
    """
    reorder a list such that the index give is that front
    :param list: list ot reorder
    :param index: new start index
    :return: reordered list
    """
    new_list = []
    for i in range(index, -1, -1):
        new_list.append(list[i])
    for i in range(len(list)-1, index, -1):
        new_list.append(list[i])
    return new_list

def make_vectors(img, keypoints, idx):
    """
    draw the 1) the gradient magnitude , 2) the gradient orientation as vectors,
    3) a 2D Gaussian weighted version of the gradient magnitude
    :param img: original image
    :param keypoints: list keypoints of type Siftpoint()
    :param idx: idx of point you want draw
    :return: None
    """

    kp = keypoints[idx]
    print(keypoints[idx].sigma, "scale!!!")

    U = np.zeros(kp.n_gradient_o.shape)
    V = np.zeros(kp.n_gradient_o.shape)

    X = np.arange(0, 16, 1)
    Y = np.arange(0, 16, 1)

    for i in range(kp.n_gradient_o.shape[0]):
        for j in range(kp.n_gradient_o.shape[1]):
            point = pol_to_cart(1, kp.n_gradient_o[i,j])
            U[i,j] = point[0]
            V[i,j] = point[1]

    gussian = create_2d_gaussian_matrix(4, 17)
    gussian_trim = gussian[0:-1,0:-1]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0][0].set_title("gradient orientation")
    q = ax[0][0].quiver(X, Y, U, V)
    ax[0][0].quiverkey(q, X=0.3, Y=1.1, U=1,
                       label='Quiver key, length = 1', labelpos='E')
    ax[0][1].set_title("Gaussian weighted gradient magnitude")
    caxes_1 = ax[0][1].imshow(kp.n_gradient_g_m)
    fig.colorbar(caxes_1, ax=ax[0][1])
    ax[1][0].set_title('gradient magnitude')
    caxes_2 = ax[1][0].imshow(kp.n_gradient_m)
    fig.colorbar(caxes_2, ax=ax[1][0])
    ax[1][1].set_title("keypoint({}, {}) on the image".format(kp.y * 2 ** kp.sigma, kp.x * 2 ** kp.sigma))
    c = plt.Circle((kp.y * 2 ** kp.sigma, kp.x * 2 ** kp.sigma), 1, color='red', linewidth=1, fill=True)
    ax[1][1].add_patch(c)
    ax[1][1].imshow(img, cmap='gray')
    plt.show()


def pol_to_cart(rho, phi):
    """
    convert the polar coordinates to cartesian coordinates
    :param rho: radius of the polar coordinate
    :param phi: angle in radians of the polar coordinate
    :return: x and y
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def plot_keypoints(keypoints, img_gray_scale ):
    """
    plots a set of keypoint on to an image using matplotlib
    :param keypoints: list of tuples of type Point to plot
    :param img_gray_scale: the image to plot the points on
    :return: None
    """
    fig,ax = plt.subplots(1)
    plt.imshow(img_gray_scale, cmap='gray')
    plt.title("key-points ")

    colors = ["blue", "green", "yellow", "magenta", "red"]

    for p in keypoints:
        c = plt.Circle((p.y * 2 **p.sigma , p.x * 2 **p.sigma), 1, color=colors[p.sigma], linewidth=1, fill=True)
        ax.add_patch(c)

    plt.show()

def filter_edge_keypoints(DoGs, cur_keypoints, plot=False):
    """
    using a hessian matrix detect corner in the difference of Gaussian and filter out keypoints that aren't in the
    detected corners
    :param DoGs: list of difference of Gaussian
    :param cur_keypoints: list of keypoints to filter
    :param plot: plot the new keypoints
    :return: filtered list of keypoints
    """

    all_keypoints = []

    pyramid_len = len(DoGs) -1
    if (len(DoGs) < 6):
        pyramid_len = pyramid_len + 1

    for i in range(pyramid_len) :
        if (plot):
            fig, ax = plt.subplots(1)
            plt.imshow(DoGs[i], cmap='gray')
            plt.title("key-points ")

        # use the conrer from the DoGs image as refrence to filter out the edge responses.
        keypoints = corner_peaks(corner_harris(DoGs[i]), min_distance=1, threshold_rel=0.05)

        for p in keypoints:

            new_pt = Point(p[0], p[1], i)


            for cur_points in cur_keypoints:
                x, y = cur_points.x, cur_points.y
                x0, y0 = new_pt.x, new_pt.y

                if ((x > x0 + 2 or x < x0 - 2) or (y > y0 + 2 or y < y0 - 2)):
                    continue

                all_keypoints.append(cur_points)
                if (plot):
                    c = plt.Circle((y, x), 1, color='red', linewidth=1, fill=True)
                    ax.add_patch(c)

        if (plot):
            plt.show()

    print("after filter_edge_keypoints", len(all_keypoints))

    return all_keypoints

if __name__ == "__main__":

    # pasre args
    parser = argparse.ArgumentParser(description='CSC 420 A1 Q4')
    parser.add_argument('img1', type=str, help='file for image 1')
    parser.add_argument('idx', type=int, help='idx of the point you want to display')
    args = parser.parse_args()

    # get images
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)

    res = image_pyramid(img1, 7)
    res_DoG = DoG(res)


    keypoints = find_keypoints(res_DoG)
    keypoints = filter_edge_keypoints(res_DoG, keypoints)

    dx_of_image_pyramid, dy_of_image_pyramid = get_dx_dy_of_image_pyramid(res)

    gradient_data = get_gradient_data(keypoints, dx_of_image_pyramid, dy_of_image_pyramid)

    make_vectors(img1, gradient_data, args.idx)

    #plot_keypoints(keypoints, img1)
    



