
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Import modules defined in the 'utils' folder
from utils import utils

# Import the models defined in the homonym file
import models




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

        if detections[0] is None: 
            num_of_detects = 0
        else:
            num_of_detects = len(detections[0])
        print("\nFinal number of detected objetcs: ", num_of_detects)

    return detections[0]


#_________________________________________________________________________



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





# Load the image that will be analyzed
img_path = "images/venice.jpg"
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Perform object detection in the image and 
# measure the time needed to complete the task
prev_time = time.time()
#img = Image.open(img_path)
img = Image.fromarray(img)
detections = detect_image(img, model)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
print ('Inference Time: ', inference_time)



# Get bounding-box colors (one for each type of object)
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# Store the image in a plt figure
img = np.array(img)
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