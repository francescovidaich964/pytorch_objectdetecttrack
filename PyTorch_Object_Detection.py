
import os, sys, time, datetime, random
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Import modules defined in the 'utilities' folder
from utilities import utils

# Import the models defined in the homonym file
import models




# Load the image that will be analyzed
img_path = "images/venice.jpg"




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



# Perform object detection in the image and 
# measure the time needed to complete the task
prev_time = time.time()
pil_img = Image.open(img_path)
detections = utils.detect_image(pil_img, model_info)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
print ('\nInference Time: ', inference_time)



# Get bounding-box colors (one for each type of object)
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# Store the image in a plt figure
img = np.array(pil_img)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)

# Define the padding transformation from/to the original image size
pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x



# Draw over the image the boxes (if any) obtained from the 'detect_image' function
if detections is not None:

    # Assign at random a color for each detected class
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)

    # For each detection, draw the corresponding bounding 
    # box and write over it the detected class of the object
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                 bbox={'color': color, 'pad': 0})


# Save the image
plt.axis('off')
plt.savefig(img_path.replace(".jpg", "-det.jpg"), bbox_inches='tight', pad_inches=0.0)