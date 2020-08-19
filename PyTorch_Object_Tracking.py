
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2

# Import modules defined in the 'utils' folder
from utils import utils

# Import the models defined in the homonym file
import models
from sort import *





def detect_image(img, model, log=False):
    """ 
    Basic function that will return detections for a specified image. 
    Note that it requires a Pillow image as input. The actual detection 
    is in the last 4 lines.
    """

    # Scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
        transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                       (128,128,128)),
        transforms.ToTensor(),
        ])

    # Convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # Run inference on the model and get detections (the model returns many detections, 
    # which will be filtered by the 'non_max_suppression' function)
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)

        if log==True:
            if detections[0] is None: 
                num_of_detects = 0
            else:
                num_of_detects = len(detections[0])
            print("\nFinal number of detected objetcs: ", num_of_detects)

    return detections[0]





##########  DETECTION MODEL INITIALIZATION  ##########


# Define paths to the YOLOv3 trained model 
# (Check on YOLO theory what are image size, confidence 
#  threshold and non-maximum suppression threshold)
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
img_size = 416
conf_thres = 0.5   # 0.8
nms_thres = 0.4    # 0.4

# Load YOLOv3 object detection model. the image will be resized to 
# a 416px square while maintaining its aspect ratio and padding the overflow. 
model = models.Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.eval()
classes = utils.load_classes(class_path)

# If CUDA is available store the model in the GPU
if torch.cuda.is_available():
    model.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

######################################################



# Define path of the input video and of the output images
videopath = '../data/full_MOT16-01.mp4'
results_path = 'results/det_image_'


# Get bounding-box colors (one for each type of object)
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# Initialize Sort object and video capture of the input video
vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()

# Analyze each image of the input video
#while(True):
for ii in range(40):

    # Take next frame of the video and get detections
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    # Define the padding transformation from/to the original image size
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Draw over the image the boxes (if any) obtained from the 'detect_image' function
    if detections is not None:

        # 'update' function of SORT object returns references to the detected 
        # objects in the image, adding to each bounding box an object ID
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    fig=plt.figure(figsize=(12, 8))
    plt.title("Video Stream")
    plt.imshow(frame)
    plt.savefig(results_path+str(ii))
