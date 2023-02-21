import cv2
import glob
import json
import os
import shutil

import numpy as np

from argparse import ArgumentParser
from bvh_skeleton import h36m_skeleton, openpose_skeleton
from pose_estimator_3d import estimator_3d
from utils import camera, smooth, vis

if __name__ == "__main__":

    parser = ArgumentParser(
        description="A Python script to convert openpose format annots to bvh.",
        epilog="python demo.py",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="EasyMocap openpose extract_keypoints saved in tmp",
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Video frames path",
    )
    args = parser.parse_args()

    ##########################
    #
    # Extract 2D keypoints from files
    #
    ##########################
    kpts_files = glob.glob(os.path.join(args.input, "*_0_keypoints.json"))
    assert len(kpts_files) > 1, "Error: Not enough files!"

    keypoints_list = {}
    for kpts_file in kpts_files:
        with open(kpts_file) as f:
            file_data = json.load(f)
            if file_data["people"] != []:
                kpts = np.array(file_data["people"][0]["pose_keypoints_2d"]).reshape(
                    (25, 3)
                )
                keypoints_list[kpts_file] = kpts

    keypoints_vals = smooth.filter_missing_value(
        keypoints_list=keypoints_list.values(),
        method="ignore",  # interpolation method will be implemented later
    )
    pose2d = np.stack(keypoints_vals)[:, :, :2]
    ##########################
    #
    # Visualize 3D pose
    #
    ##########################
    vis_result_dir = "2d_pose_vis"  # path to save the visualized images
    if os.path.exists(vis_result_dir):
        shutil.rmtree(vis_result_dir)
    os.makedirs(vis_result_dir)

    op_skel = openpose_skeleton.OpenPoseSkeleton()

    for file_path in keypoints_list:

        basename = os.path.basename(file_path).split("_")[0]
        frame_path = os.path.join(args.video, basename + ".jpg")
        if os.path.exists(frame_path):
            # keypoint whose detect confidence under kp_thresh will not be visualized
            vis.vis_2d_keypoints(
                keypoints=keypoints_list[file_path],
                img=cv2.imread(frame_path),
                skeleton=op_skel,
                kp_thresh=0.4,
                output_file=os.path.join(vis_result_dir, basename + ".png"),
            )

    ##########################
    #
    # Estimate 3D pose from 2D pose
    #
    ##########################
    e3d = estimator_3d.Estimator3D(
        config_file="models/openpose_video_pose_243f/video_pose.yaml",
        checkpoint_file="models/openpose_video_pose_243f/best_58.58.pth",
    )
    pose3d = e3d.estimate(pose2d, image_width=568, image_height=320)

    # subject = "S1"
    # cam_id = "55011271"
    # cam_params = camera.load_camera_params("cameras.h5")[subject][cam_id]
    # R = cam_params["R"]
    # T = 0
    # azimuth = cam_params["azimuth"]

    pose3d_world = camera.camera2world(pose=pose3d, R=np.eye(3), T=0)
    pose3d_world[:, :, 2] -= np.min(pose3d_world[:, :, 2])  # rebase the height

    ##########################
    #
    # Visualize 3D pose
    #
    ##########################
    h36m_skel = h36m_skeleton.H36mSkeleton()
    gif_file = "3d_pose.gif"  # output format can be .gif or .mp4

    ani = vis.vis_3d_keypoints_sequence(
        keypoints_sequence=pose3d_world[0:300],
        skeleton=h36m_skel,
        elev=-90,
        azimuth=-90,  # azimuth,
        fps=60,
        output_file=gif_file,
    )

    ##########################
    #
    # Convert 3D pose to BVH
    #
    ##########################
    bvh_file = "test.bvh"
    _, _ = h36m_skel.poses2bvh(pose3d_world, output_file=bvh_file)
