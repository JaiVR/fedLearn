import os
import torch
import cv2
import numpy as np

from server import load_model
from yolov5.utils.loss import ComputeLoss
from yolov5.val import run

net = load_model()
# print(net)
loss_fn = None #ComputeLoss(net)

print("Evaluating model...")
out = run(data='yolov5/data/coco128.yaml', weights="yolov5/weights/yolov5n.pt", compute_loss=loss_fn)
print(out[0][4:])