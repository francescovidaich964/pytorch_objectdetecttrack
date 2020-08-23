
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim 
import numpy as np
from PIL import Image
import cv2

# Import modules defined in the 'utils' folder
from utils import utils

# Import the models defined in the homonym file
import models
from sort import *

import imageio
from tqdm import tqdm



# Define path of the input and output video frames
#input_path  = '../InOut_data/MOT16/images/train/MOT16-04/img1/'
input_path  = '../InOut_data/data_test/'
output_video_path = '../InOut_data/FFMpegWriter_test.mp4'






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




# Collect names of all images of the input video
in_image_names = os.listdir(path=input_path)
in_image_names.sort()

# Get bounding-box colors (one for each type of object) 
#cmap = plt.get_cmap('tab20b')
#colors = [cmap(i) for i in np.linspace(0, 1, 20)]  (BETTER TO HAVE COLORS FROM DIFFERENT CMAPS)
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),
        (0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]


# Initialize Sort object
mot_tracker = Sort()

# Initialize the writer which will convert the frames into the video
writer = anim.FFMpegWriter(fps=25, codec='mpeg4', bitrate=2000) 
plt.figure() 
fig, ax = plt.subplots(1, figsize=(12,9))


with writer.saving(fig, output_video_path, dpi=100): 
    # Analize each frame of the input video
    for img_name in tqdm(in_image_names):

        # Take next frame of the video and get detections
        #ret, frame = vid.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #pilimg = Image.fromarray(frame)
        pil_img = Image.open(input_path + img_name)
        detections = detect_image(pil_img, model)

        # Store the image in a plt figure    
        img = np.array(pil_img)
        #plt.figure()
        #fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(img)

        # Define the padding transformation from/to the original image size
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        # Draw over the image the boxes (if any) obtained from the 'detect_image' function
        if detections is not None:

            # 'update' function of SORT object returns references to the detected 
            # objects in the image, adding to each bounding box an object ID
            tracked_objects = mot_tracker.update(detections.cpu())

            # Assign at random a color for each detected class
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            # For each tracked object (charachterized by the obj_id returned from SORT), draw 
            # the corresponding bounding box and write over it the detected class of the object
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = colors[int(obj_id) % len(colors)]
                color = [i / 255 for i in color]
                class_name = classes[int(cls_pred)]

                # OLD VERSION WITH OPEN-CV
                #cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                #cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                #cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(x1, y1, s=class_name, color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})


        # Save the current frame with boxes
        plt.axis('off')
        fig.tight_layout()
        writer.grab_frame() 
        plt.cla()
        #img_out_name = output_path + img_name.replace(".jpg", "-det.jpg")
        #plt.savefig(img_out_name, bbox_inches='tight', pad_inches=0.0)
