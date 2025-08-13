import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def run_webcam(cam_id, model, device, imgsz, half, opt):
    """Run detection on a webcam device."""
    cap = cv2.VideoCapture(cam_id)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print(f'Starting webcam device {cam_id}... Press q to quit.')
    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (imgsz, imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        for det in pred:
            s = ''
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
        cv2.imshow(f'Webcam {cam_id}', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_stream(source, model, device, imgsz, half, opt):
    """Run detection on a video stream, file, or URL."""
    cudnn.benchmark = True
    dataset = LoadStreams(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print(f'Starting stream {source}... Press q to quit.')
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):
                raise StopIteration

def detect_realtime():
    """Main entry point for real-time detection."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='2', help='source, e.g. 0 for webcam or rtsp/http/video path')
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    source = opt.source
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(opt.weights, map_location=device)
    imgsz = check_img_size(opt.img_size, s=model.stride.max())
    if half:
        model.half()

    # Decide input type
    if source.isdigit():
        run_webcam(int(source), model, device, imgsz, half, opt)
    else:
        run_stream(source, model, device, imgsz, half, opt)

if __name__ == '__main__':
    with torch.no_grad():
        detect_realtime()
