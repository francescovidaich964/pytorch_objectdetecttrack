
import os, sys, time, datetime, random
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim 
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import modules defined in the 'utilities' folder
from utilities import utils

# Import the models defined in the homonym file
import models
from sort import *




####### (EDIT HERE) Define path of the input and output video frames #######

input_path  = '../InOut_data/data_test/'
output_video_path = '../InOut_data/tracked_video.mp4'

############################################################################




################## DETECTION MODEL INITIALIZATION ##################

# Define paths to the YOLOv3 trained model 
# (Check on YOLO theory what are image size, confidence 
#  threshold and non-maximum suppression threshold)
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
img_size = 416
conf_thres = 0.6   # 0.8
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

# Pack all these information in a single list
model_info = [model, img_size, conf_thres, nms_thres, Tensor]

####################################################################




#__________________________________________________________________________________




# Collect names of all images of the input video
in_image_names = os.listdir(path=input_path)
in_image_names.sort()

# Get bounding-box colors (one for each type of object) 
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),
        (0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]


# Initialize Sort object
mot_tracker = Sort()

# Initialize the writer object (converts frames into the video)
writer = anim.FFMpegWriter(fps=25, codec='mpeg4', bitrate=5000) 
plt.figure() 
fig, ax = plt.subplots(1, figsize=(12.8,7.2), dpi=100)




# Open the stream to the output video, which will be built frame by frame
with writer.saving(fig, output_video_path, dpi=100): 
    
    # Analize each frame of the input video
    for img_name in tqdm(in_image_names):

        # Take next frame of the video and get detections
        pil_img = Image.open(input_path + img_name)
        detections = utils.detect_image(pil_img, model_info)

        # Store the image in a plt figure    
        img = np.array(pil_img)
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

            # Assign a color for each detected object
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

                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(x1, y1, s=class_name, color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})


        # Save the current frame with boxes inside the video
        plt.axis('off')
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        writer.grab_frame() 
        plt.cla()   # clear axis
        #img_out_name = output_path + img_name.replace(".jpg", "-det.jpg")
        #plt.savefig(img_out_name, bbox_inches='tight', pad_inches=0.0)
