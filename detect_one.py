'''
@author: 刘家兴
@contact: 1445101363@qq.com
@file: detect_one.py
@time: 2022/8/16 15:40
@desc:
'''
# -*- coding: UTF-8 -*-
import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
import math

from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression_landmark, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.yolo import Model
import yaml

import warnings

warnings.filterwarnings("ignore")


def load_model(weights, cfg_path, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model = Model(cfg_path, ch=3, nc=1 ).to(device)  # create
    # ckpt = torch.load(weights, map_location='cpu')  # load
    # state_dict = ckpt['model'].float().state_dict()
    # model.load_state_dict(state_dict)
    # model.eval()
    return model


def get_img(input_dir):
    xml_path_list = []
    for (root_path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                xml_path = root_path + "/" + filename
                xml_path_list.append(xml_path)
    return xml_path_list


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords


def get_ther(landmarkspoints, class_num):
    class_num = int(class_num)
    if class_num > 5:
        pMax = (
        (landmarkspoints[0][0] + landmarkspoints[1][0]) / 2, (landmarkspoints[0][1] + landmarkspoints[1][1]) / 2)
        pMin = (
        (landmarkspoints[2][0] + landmarkspoints[3][0]) / 2, (landmarkspoints[2][1] + landmarkspoints[3][1]) / 2)
    elif class_num > 1:
        pMin = (
        (landmarkspoints[0][0] + landmarkspoints[1][0]) / 2, (landmarkspoints[0][1] + landmarkspoints[1][1]) / 2)
        pMax = (
        (landmarkspoints[2][0] + landmarkspoints[3][0]) / 2, (landmarkspoints[2][1] + landmarkspoints[3][1]) / 2)

    # x, y = pMax[0] - pMin[0], pMin[1] - pMax[1]
    x, y = pMin[0] - pMax[0], pMax[1] - pMin[1]
    ther = math.atan2(x, y)

    return int(ther / math.pi * 180)


def show_results(img, xywh, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 2 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    landmarkspoints = []
    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
        landmarkspoints.append([point_x, point_y])

    mask = np.zeros((h, w, c), dtype=np.uint8)

    cv2.fillConvexPoly(mask, np.array(landmarkspoints), (0, 100, 255))  # 绘制 地面投影
    img = cv2.addWeighted(img, 1, mask, 0.5, 0)

    ther = get_ther(landmarkspoints, class_num)

    tf = max(tl - 1, 1)  # font thickness
    # label = str(int(class_num)) + ': ' + str(conf)[:5] + '&:' +str(ther)
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    print('label', str(int(class_num)) + ': ' + str(conf)[:5] + ' 航向角:' + str(ther))
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def warpimage(img, pts1):
    # right_bottom, left_bottom, left up , left_up
    # pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[400, 200], [0, 200], [0, 0], [400, 0]])
    h, w, c = img.shape
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (400, 200))
    return dst


def detect_one(model, image_path, device, output_dir):
    # Load model
    img_size = 320
    # img_size = 416
    conf_thres = 0.5
    iou_thres = 0.45

    save_path = os.path.join(output_dir, os.path.split(image_path)[1])

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]
    print('pred: ', pred.shape)
    # Apply NMS
    # pred = non_max_suppression_landmark(pred, conf_thres, iou_thres)
    pred = non_max_suppression_landmark(pred, conf_thres, iou_thres, multi_label=True)
    # print('nms: ', pred)
    t2 = time_synchronized()

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 14:22] = scale_coords_landmarks(img.shape[2:], det[:, 14:22], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(torch.tensor(det[j, :4]).view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 14:22].view(1, 8) / gn_lks).view(-1).tolist()
                class_num = det[j, 22].cpu().numpy()

                orgimg = show_results(orgimg, xywh, conf, landmarks, 7)

    # Stream results
    print(f'Done. ({time.time() - t0:.3f}s)')

    # cv2.imshow('orgimg', orgimg)
    print(save_path)
    cv2.imwrite(save_path, orgimg)
    # if cv2.waitKey(0) == ord('q'):  # q to quit
    #    raise StopIteration


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--yaml', type=str, default="")
    parser.add_argument('--input', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.model
    cfg_path = args.yaml
    model = load_model(weights, cfg_path, device)

    # root = '/home/xialuxi/work/dukto/data/CCPD2020/CCPD2020/images/test/'
    img_path_list = get_img(args.input)
    # image_path = '/mnt/sdb2/dataset/keypoint2/JPEGImages/2022_05_23_15_43_30_1653320610624.jpg'
    # image_path = '/home/wqg/data/maxvision_data/ADAS/1964/train/images/0016.jpg'
    for image_path in tqdm(img_path_list):
        detect_one(model, image_path, device, args.output_dir)
    print('over')


