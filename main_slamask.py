"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import math
import time

import platform

from config import Config

from slam import Slam, SlamState
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

if platform.system()  == 'Linux':
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters
import multiprocessing as mp


if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)
    #groundtruth = None # not actually used by Slam() class; could be used for evaluating performances

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])

    num_features=2000

    tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn
    #tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching

    # select your tracker configuration (see the file feature_tracker_configs.py)
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT, CONTEXTDESC
    tracker_config = FeatureTrackerConfigs.TEST
    tracker_config['num_features'] = num_features
    tracker_config['tracker_type'] = tracker_type

    print('tracker_config: ',tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create SLAM object
    slam = Slam(cam, feature_tracker, groundtruth)
    time.sleep(1) # to show initial messages

    is_draw_err = True
    err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')
    err_norm_plt = Mplot2d(xlabel='img id', ylabel='norm',title='error_norm')
    err_rot_plt = Mplot2d(xlabel='img id', ylabel='Deg. per meter',title='Rotation error')

    viewer3D = Viewer3D()

    if platform.system()  == 'Linux':
        display2d = Display2D(cam.width, cam.height)  # pygame interface
    else:
        display2d = None  # enable this if you want to use opencv window

    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    do_step = False
    is_paused = False

    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed
    while dataset.isOk():

        if not is_paused:
            print('..................................')
            print('image: ', img_id)
            img = dataset.getImageColor(img_id)
            mask = dataset.getMask(img_id)
            if img is None:
                print('image is empty')
                getchar()
            timestamp = dataset.getTimestamp()          # get current timestamp
            next_timestamp = dataset.getNextTimestamp() # get next timestamp
            frame_duration = next_timestamp-timestamp

            if img is not None:
                time_start = time.time()
                print("main", mask.shape)
                slam.track(img, img_id, timestamp, mask=mask)  # main SLAM function

                if len(slam.tracking.traj3d_est) > 0:
                    is_draw_err = True
                    is_draw_err_rot = True
                    x, y, z = slam.tracking.traj3d_est[-1]
                    x_true, y_true, z_true = slam.tracking.traj3d_gt[-1]

                    rot = slam.tracking.trajrot_est[-1]
                    rot_true = slam.tracking.trajrot_gt[-1]
                else:
                    is_draw_err = False
                    is_draw_err_rot = False


                # 3D display (map display)
                if viewer3D is not None:
                    viewer3D.draw_map(slam)

                img_draw = slam.map.draw_feature_trails(img)

                # 2D display (image display)
                if display2d is not None:
                    display2d.draw(img_draw)
                else:
                    cv2.imshow('Camera', img_draw)

                if matched_points_plt is not None:
                    if slam.tracking.num_matched_kps is not None:
                        matched_kps_signal = [img_id, slam.tracking.num_matched_kps]
                        matched_points_plt.draw(matched_kps_signal,'# keypoint matches',color='r')
                    if slam.tracking.num_inliers is not None:
                        inliers_signal = [img_id, slam.tracking.num_inliers]
                        matched_points_plt.draw(inliers_signal,'# inliers',color='g')
                    if slam.tracking.num_matched_map_points is not None:
                        valid_matched_map_points_signal = [img_id, slam.tracking.num_matched_map_points]   # valid matched map points (in current pose optimization)
                        matched_points_plt.draw(valid_matched_map_points_signal,'# matched map pts', color='b')
                    if slam.tracking.num_kf_ref_tracked_points is not None:
                        kf_ref_tracked_points_signal = [img_id, slam.tracking.num_kf_ref_tracked_points]
                        matched_points_plt.draw(kf_ref_tracked_points_signal,'# $KF_{ref}$  tracked pts',color='c')
                    if slam.tracking.descriptor_distance_sigma is not None:
                        descriptor_sigma_signal = [img_id, slam.tracking.descriptor_distance_sigma]
                        matched_points_plt.draw(descriptor_sigma_signal,'descriptor distance $\sigma_{th}$',color='k')
                    matched_points_plt.refresh()

                if is_draw_err:         # draw error signals
                    errx = [img_id, math.fabs(x_true-x)]
                    erry = [img_id, math.fabs(y_true-y)]
                    errz = [img_id, math.fabs(z_true-z)]
                    err_plt.draw(errx,'err_x',color='g')
                    err_plt.draw(erry,'err_y',color='b')
                    err_plt.draw(errz,'err_z',color='r')
                    err_plt.refresh()

                    # L2 norm error
                    norm = (x_true - x) **2 + (y_true - y) **2 + (z_true - z) **2
                    err_norm_plt.draw([img_id, norm**0.5], 'L2 norm', color='r')
                    err_norm_plt.refresh()

                if is_draw_err_rot:
                    rot_error = np.linalg.inv(rot) @ rot_true

                    a = rot_error[0, 0]
                    b = rot_error[1, 1]
                    c = rot_error[2, 2]
                    d = 0.5*(a+b+c-1.0)
                    err = np.arccos(max(min(d, 1.0), -1.0)) # KITTI BENCHMARK devkit codebase | Ref: https://www.cvlibs.net/datasets/kitti/eval_odometry_detail.php?&result=fb25c982b8f6294ef06efd6baa96f0178f35cbc7
                    err_rot_plt.draw([img_id, err], 'degree per meter error', color='r')
                    err_rot_plt.refresh()


                duration = time.time()-time_start
                if(frame_duration > duration):
                    print('sleeping for frame')
                    time.sleep(frame_duration-duration)

            img_id += 1
        else:
            time.sleep(1)

        # get keys
        key = matched_points_plt.get_key()
        key_cv = cv2.waitKey(1) & 0xFF

        # manage interface infos

        if slam.tracking.state==SlamState.LOST:
            if display2d is not None:
                getchar()
            else:
                key_cv = cv2.waitKey(0) & 0xFF   # useful when drawing stuff for debugging

        if do_step and img_id > 1:
            # stop at each frame
            if display2d is not None:
                getchar()
            else:
                key_cv = cv2.waitKey(0) & 0xFF

        if key == 'd' or (key_cv == ord('d')):
            do_step = not do_step
            Printer.green('do step: ', do_step)

        if key == 'q' or (key_cv == ord('q')):
            if display2d is not None:
                display2d.quit()
            if viewer3D is not None:
                viewer3D.quit()
            if matched_points_plt is not None:
                matched_points_plt.quit()
            break

        if viewer3D is not None:
            is_paused = not viewer3D.is_paused()

    slam.quit()

    #cv2.waitKey(0)
    cv2.destroyAllWindows()
