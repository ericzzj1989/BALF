#!/usr/bin/env python
# coding: utf-8

# Imports
import numpy as np
import cv2
import time


class MatcherWrapper(object):
    """OpenCV matcher wrapper."""

    def __init__(self, descr1, descr2, feature, matcher, ratio=None, cross_check=True):
        """
        Args:
            feat: (n_kpts, 128) Local features.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
        Returns:
            good_matches: Putative matches.
        """

        if matcher == 'BF':
            if (feature == 'ORB') or (feature == 'BRISK'):
                normType = cv2.NORM_HAMMING
            else:
                normType = cv2.NORM_L2
            matcher = cv2.BFMatcher(normType=normType, crossCheck = True)
            matches = matcher.match(queryDescriptors = descr1, trainDescriptors = descr2)
            self.good_matches = sorted(matches, key = lambda x: x.distance)
        
        elif matcher == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            # matcher = cv2.BFMatcher(cv2.NORM_L2)
            if (feature == 'ORB') or (feature == 'BRISK') or (feature == 'DISK'):
                descr1 = np.float32(descr1)
                descr2 = np.float32(descr2)
            init_matches1 = matcher.knnMatch(descr1, descr2, k=2)
            init_matches2 = matcher.knnMatch(descr2, descr1, k=2)

            self.good_matches = []

            for i in range(len(init_matches1)):
                cond = True
                # Mutual nearest neighbor constraint
                if cross_check:
                    cond1 = cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i
                    cond *= cond1
                # Lowe’s ratio test
                if ratio is not None and ratio < 1:
                    cond2 = init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance
                    cond *= cond2
                if cond:
                    self.good_matches.append(init_matches1[i][0])

    def get_matches(self, cv_kpts1, cv_kpts2, err_thld=4, ransac=True, info=''):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_thld: Epipolar error threshold.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """

        if type(cv_kpts1) is list and type(cv_kpts2) is list:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in self.good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in self.good_matches])
        elif type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx] for m in self.good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx] for m in self.good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        t0 = time.time()
        if ransac:
            F_matrix, mask = cv2.findFundamentalMat(
                good_kpts1, good_kpts2, cv2.RANSAC, err_thld, 0.999
            )
            n_inliers = np.count_nonzero(mask)
            print(info, 'n_putative', len(self.good_matches), 'n_inliers', n_inliers)
        else:
            mask = np.ones((len(self.good_matches), ))
            print(info, 'n_putative', len(self.good_matches))
        t1 = time.time()
        time_ransac = t1-t0

        return self.good_matches, mask, time_ransac, n_inliers, F_matrix

    def get_inliers(self, cv_kpts1, cv_kpts2, err_thld=4, ransac=True, info=''):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_thld: Epipolar error threshold.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """

        if type(cv_kpts1) is list and type(cv_kpts2) is list:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in self.good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in self.good_matches])
        elif type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx] for m in self.good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx] for m in self.good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        corres1 = []
        corres2 = []

        t0 = time.time()
        if ransac:
            F_matrix, mask = cv2.findFundamentalMat(
                good_kpts1, good_kpts2, cv2.RANSAC, err_thld, 0.999
            )
            n_inliers = np.count_nonzero(mask)
            print(info, 'n_putative', len(self.good_matches), 'n_inliers', n_inliers)

            idx_inliers = np.asarray(np.where(mask==1))[0,:]
            for i, idx_inlier in enumerate(idx_inliers):
                corres1.append(good_kpts1[idx_inlier, :])
                corres2.append(good_kpts2[idx_inlier, :])

            corr1 = np.asarray(corres1)
            corr2 = np.asarray(corres2)
            
            # print('**************************')
            # print('sharpA corres: ', corr1.shape)
            # print('**************************')
            # print('sharpB corres: ', corr2.shape)
            # print('**************************')
        else:
            mask = np.ones((len(self.good_matches), ))
            print(info, 'n_putative', len(self.good_matches))
        t1 = time.time()
        time_ransac = t1-t0

        return self.good_matches, mask, time_ransac, n_inliers, F_matrix, corr1, corr2

    def draw_matches(self, image1_path, cv_kpts1, image2_path, cv_kpts2, good_matches, mask, 
                     match_color=(0, 255, 0), pt_color=(0, 255, 100)):
        """Draw matches."""
        img1 = cv2.imread(image1_path) 
        img2 = cv2.imread(image2_path)
        if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1)
                        for i in range(cv_kpts1.shape[0])]
            cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1)
                        for i in range(cv_kpts2.shape[0])]
        display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                  None,
                                  matchColor=match_color,
                                  singlePointColor=pt_color,
                                  matchesMask=mask.ravel().tolist(),
                                  flags=4)
        return display