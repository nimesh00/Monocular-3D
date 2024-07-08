'''
ALL THE CODE WRITTEN IN THIS FILE HAS BEEN REFERENCED ONLY FROM THE SOURCE DOCUMENTATION AND IS WRITTEN COMPLETELY BY ME. NO PART OF IT IS COPIED FROM SOMEONE ELSE'S WORK.
'''

import cv2
import numpy as np
# import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *

def main():
    # print("something")
    images, intrinsic_matrices, transformation_matrices = read_data(6)

    '''Reference from: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html'''

    # window_size = 7
    block_size = 8
    min_disp = -15
    max_disp = 128
    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        P1 = 8*1*block_size**2,
        P2 = 32*2*block_size**2,
        disp12MaxDiff = 2,
        uniquenessRatio = 5,
        speckleWindowSize = 300,
        speckleRange = 1
    )

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    net_transformation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    for i in range(13, 15):
        net_transformation = np.linalg.inv(transformation_matrices[i])
        camera_pose = net_transformation @ np.reshape(np.array([0, 0, 0, 1]), (4, 1))
        print(np.linalg.norm(camera_pose[:3]))
        # chaning the colorspace to get a single channel
        img1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)
        # _, img1 = cv2.threshold(img1, 20, 255, cv2.THRESH_TOZERO)
        # _, img2 = cv2.threshold(img2, 20, 255, cv2.THRESH_TOZERO)
        plt.subplot(2, 1, 1)
        plt.imshow(img1)
        plt.subplot(2, 1, 2)
        plt.imshow(img2)
        plt.show()
        # img1 = img1[50:-50, 50:-50]
        # img2 = img2[50:-50, 50:-50]

        kp1, des1, kp2, des2, matches = detect_and_match_features(img1, img2, sift, flann)
        matchesMask, pts1, pts2 = lowes_ratio_test(matches, kp1, kp2, 0.5)

        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)

        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

        # plt.imshow(img3),plt.show()

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]


        imgsize = (img1.shape[1], img1.shape[0])
        _, h1, h2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgsize)
        T_12 = transformation_matrices[i] @ np.linalg.inv(transformation_matrices[i + 1])
        # rev_proj_matrix = np.zeros((4,4))
        # R1, R2, P1, P2, Q, vROI1, vROI2 = cv2.stereoRectify(cameraMatrix1 = intrinsic_matrices[i], distCoeffs1 = 0, cameraMatrix2 = intrinsic_matrices[i + 1], distCoeffs2 = 0, imageSize = imgsize, R = T_12[:3, :3], T = T_12[:3, 3], Q = rev_proj_matrix)
        # f = intrinsic_matrices[i][0, 0]
        # h, w = img1.shape
        # Q = np.float32([[1, 0, 0, -0.5*w],
        #             [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
        #             [0, 0, 0,     -f], # so that y-axis looks up
        #             [0, 0, 1,      0]])
        # print(Q)
        # print(h1)
        # print(h2)
        rectified1 = cv2.warpPerspective(img1, h1, imgsize)
        rectified2 = cv2.warpPerspective(img2, h2, imgsize)
        # plt.subplot(2, 1, 1)
        # plt.imshow(rectified1)

        # plt.subplot(2, 1, 2)
        # plt.imshow(rectified2)

        # plt.show()

        disparity = stereo.compute(rectified1, rectified2)
        disparity = cv2.normalize(disparity, disparity, alpha = 255, beta = 0, norm_type = cv2.NORM_MINMAX)
        disparity = np.uint8(disparity)

        disparity_unwarped = cv2.warpPerspective(disparity, np.linalg.inv(h1), imgsize)

        x,y,w,h = cv2.boundingRect(disparity_unwarped)

        disparity_unwarped = disparity_unwarped[y : y + h, x : x + w]

        plt.imshow(disparity_unwarped, 'gray'), plt.show()

        # depth_map = create_depth_from_disparity(disparity_unwarped, intrinsic_matrices[i], T_12)
        # points_3D = points_from_depth_map()

        # plt.imshow(depth_map, 'gray')
        # plt.show()
        
        
        
        pts_3d = cv2.reprojectImageTo3D(disparity_unwarped, T_12, handleMissingValues = 0)
        # print(pts_3d)
        # model_pts = [[], [], []]
        x_pts_3D = (pts_3d[:, :, 0]).flatten() / 5
        y_pts_3D = (pts_3d[:, :, 1]).flatten() / 5
        z_pts_3D = (pts_3d[:, :, 2]).flatten() / 5
        print(x_pts_3D[0:-1:100])

        camera_frame_points = np.ones((4, len(x_pts_3D)))
        camera_frame_points[0, :] = x_pts_3D
        camera_frame_points[1, :] = y_pts_3D
        camera_frame_points[2, :] = z_pts_3D

        print(len(x_pts_3D))

        global_pts_3D = np.linalg.inv(transformation_matrices[i]) @ camera_frame_points

        # print("max x,min x: ", np.max(global_pts_3D[0]), np.min(global_pts_3D[0]))
        # print("max y,min y: ", np.max(global_pts_3D[1]), np.min(global_pts_3D[1]))
        # print("max z,min z: ", np.max(global_pts_3D[2]), np.min(global_pts_3D[2]))

        # distance_field, volumetric_histogram = create_distance_field(global_pts_3D[:3, :], camera_pose[:3])



        plot_3D_points(global_pts_3D[0, 0:-1:50], global_pts_3D[1, 0:-1:50], global_pts_3D[2, 0:-1:50])

        # plot_3D_point/s(x_pts_3D, y_pts_3D, z_pts_3D)

        # plt.imshow(disparity_unwarped, 'gray')
        # plt.show()

        # # Find epilines corresponding to points in right image (second image) and
        # # drawing its lines on left image
        # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        # lines1 = lines1.reshape(-1,3)
        # img5,img6 = drawEpilines(img1,img2,lines1,pts1,pts2)

        # # Find epilines corresponding to points in left image (first image) and
        # # drawing its lines on right image
        # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        # lines2 = lines2.reshape(-1,3)
        # img3,img4 = drawEpilines(img2,img1,lines2,pts2,pts1)

        # plt.subplot(121),plt.imshow(img5)
        # plt.subplot(122),plt.imshow(img3)
        # plt.show()

        # plt.imshow(img3,), plt.show()

if __name__ == "__main__":
    main()