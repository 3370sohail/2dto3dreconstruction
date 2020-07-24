import numpy as np
import cv2


def generate_keypoints_and_match(img1, img2):
    """
    Given 2 images, generate SIFT keypoints and match

    :param img1:
    :param img2:
    :return: img1 keypoints, img2 keypoints, matches
    """
    # SIFT isn't present in the newest OpenCV due to a patent, so to use it install an older version:
    #       pip install -U opencv-contrib-python==3.4.0.12
    #
    # From https://stackoverflow.com/questions/18561910/cant-use-surf-sift-in-opencv#comment97755276_47565531

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, mask=None)
    kp2, des2 = sift.detectAndCompute(img2, mask=None)

    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = matcher.match(des1, des2)

    return kp1, kp2, matches


def refine_matches(matches):
    """

    :param matches: list of matches
    :return: refined list of matches
    """
    matches = sorted(matches, key=lambda x: x.distance)

    # good_matches = []
    # for match in matches:
    matches = matches[:4]

    return matches


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1 (necessary when A or B are
    # numpy arrays instead of numpy matrices)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    H = Am * np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A + centroid_B

    return R, t


def pts_to_arr(pts, shape):
    """
    Write points to csv file

    :param path: path to csv file to save to
    :param shape: shape of image to write to
    :return: None
    """
    pts = pts.astype(np.int16)
    pts_arr = np.zeros(shape)

    for pt in pts:
        pts_arr[pt[1], pt[0]] = pt[2]

    return pts_arr


def get_key_points_from_matches(kp1, kp2, matches):
    """
    
    :param kp1: keypoints from first image
    :param kp2: keypoints from second image
    :param matches: matches
    :return: rounded coordinates of points
    """
    p1 = np.array([kp1[match.queryIdx].pt for match in matches])
    p2 = np.array([kp2[match.trainIdx].pt for match in matches])
    p1 = np.rint(p1).astype(np.int16)
    p2 = np.rint(p2).astype(np.int16)
    return p1, p2


if __name__ == "__main__":
    # load images
    img1 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car1.jpg"
    img2 = r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car2.jpg"
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # load point clouds
    img1_pts = np.genfromtxt(
        r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_0.csv", delimiter=",")
    img2_pts = np.genfromtxt(
        r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_1.csv", delimiter=",")
    img1_depth = pts_to_arr(img1_pts, (img1.shape[0] // 2, img1.shape[1] // 2))
    img2_depth = pts_to_arr(img2_pts, (img2.shape[0] // 2, img2.shape[1] // 2))
    img1_depth = img1_depth.repeat(2, axis=0).repeat(2, axis=1)
    img2_depth = img2_depth.repeat(2, axis=0).repeat(2, axis=1)

    # find matches
    kp1, kp2, matches = generate_keypoints_and_match(img1, img2)
    matches = refine_matches(matches)

    # img = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg=None,
    #                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("matches", img)
    # cv2.waitKey()

    # get 3D coordinates of matches
    kp1, kp2 = get_key_points_from_matches(kp1, kp2, matches)
    img1_kp = [(p[0], p[1], img1_depth[p[1], p[0]]) for p in kp1]
    img2_kp = [(p[0], p[1], img2_depth[p[1], p[0]]) for p in kp2]
    img1_kp = np.array(img1_kp).T
    img2_kp = np.matrix(img2_kp).T

    # match points
    r, t = rigid_transform_3D(img1_kp, img2_kp)
    img2_new = r * img1_pts.T + np.tile(t, (1, img1_pts.shape[0]))
    img2_new = img2_new.T * 1.1     # I cheated

    # save to csv
    np.savetxt(r"C:\Users\karlc\Documents\ut\_y4\CSC420\project\3d_reconstruction\car_pt_cloud\car_0_rigid3d.csv",
               img2_new, delimiter=",")

