#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pose Estimation running score.


import numpy as np


class PoseRunningScore(object):
    def __init__(self, ):
        self.oks_all = np.zeros(0)
        self.oks_num = 0

    def compute_oks(self, gt_kpts, pred_kpts):
        # print(gt_kpts.shape)
        # print(pred_kpts.shape)
        if pred_kpts.shape[-1] ==0:
            return np.array([[1]])
        # print("gt:",gt_kpts)
        # print("pre",pred_kpts)
        """Compute oks matrix (size gtN*pN)."""
        gt_count = len(gt_kpts)
        pred_count = len(pred_kpts)
        oks = np.zeros((gt_count, pred_count))
        if pred_count == 0:
            return oks.T

        # for every human keypoint annotation
        for i in range(gt_count):
            anno_keypoints = np.reshape(np.array(gt_kpts[i]), (4, 2))

            scale = max(np.max(anno_keypoints[:, 0]) - np.min(anno_keypoints[:, 0]),
                        np.max(anno_keypoints[:, 0]) - np.min(anno_keypoints[:, 0])) ** 2 + 1e-8


            # for every predicted human
            for j in range(pred_count):
                predict_keypoints = np.reshape(np.array(pred_kpts[j]),
                                               (4,2))
                dis = np.sum((anno_keypoints - predict_keypoints) ** 2, axis=1)
                oks[i, j] = np.mean(
                    np.exp(-dis / 2 / 1 ** 2 / (scale + 1))
                )

        return oks

    def update(self, batch_pred_kpts, batch_gt_kpts,oks_all):
        """Evaluate predicted_file and return mAP."""
        # Construct set to speed up id searching.
        # for every annotation in our test/validation set
        for i in range(len(batch_pred_kpts)):
            # if the image in the predictions, then compute oks
            # oks = self.compute_oks(batch_gt_kpts[i], batch_pred_kpts[i])
            oks = oks_all[i]
            # print(oks)
            # view pairs with max OKSs as match ones, add to oks_all
            self.oks_all = np.concatenate((self.oks_all, np.max(oks, axis=1)), axis=0)
            # accumulate total num by max(g
            self.oks_num += np.max(oks.shape)

    def get_mAP(self):
        # compute mAP by APs under different oks thresholds
        average_precision = []
        # for threshold in np.linspace(0.25, 0.95, 10):
        threshold=0.25
        average_precision.append(np.sum(self.oks_all > threshold) / np.float32(self.oks_num))

        return np.mean(average_precision)

    def reset(self):
        self.oks_all = np.zeros(0)
        self.oks_num = 0

class PoseRunningScore1(object):
    def __init__(self):
        self.preds = list()
        self.gts = list()

    def update(self, ground_truth, prediction):
        if isinstance(ground_truth, list):
            self.gts += ground_truth
        else:
            self.gts += [ground_truth]

        if isinstance(prediction, list):
            self.preds += prediction
        else:
            self.preds += [prediction]

    def get_scores(self):
        """Returns the evaluation params specified in the list"""

        gt_global_poses = np.concatenate(self.gts)
        pred_poses = np.concatenate(self.preds)

        gt_global_poses = np.concatenate(
            (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
        gt_global_poses[:, 3, 3] = 1
        gt_xyzs = gt_global_poses[:, :3, 3]
        gt_local_poses = []
        for i in range(1, len(gt_global_poses)):
            gt_local_poses.append(
                np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))
        ates = []
        num_frames = gt_xyzs.shape[0]
        track_length = 5
        for i in range(0, num_frames - track_length + 1):
            local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
            gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
            ates.append(compute_ate(gt_local_xyzs, local_xyzs))

        pose_error = {'mean': np.mean(ates), 'std': np.std(ates)}
        return pose_error

    def reset(self):
        self.preds = list()
        self.gts = list()