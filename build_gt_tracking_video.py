
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim 
import numpy as np
from PIL import Image
from tqdm import tqdm





# Define input and output paths
input_path = "../InOut_data/MOT16/images/train/MOT16-04/img1/"
gt_boxes_file = "../InOut_data/MOT16/images/train/MOT16-04/gt/gt.txt"
output_video_path = '../InOut_data/gt_tracked_video.mp4'




# Load ground_truth bounding boxes from the file (remove useless data)
gt_boxes = np.loadtxt(gt_boxes_file, delimiter=',') 
gt_boxes = gt_boxes[:,:6]

# Collect names of all images of the input video
in_image_names = os.listdir(path=input_path)
in_image_names.sort()

# Defne bounding-box colors
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),
        (0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

# Initialize the writer object (converts frames into the video)
writer = anim.FFMpegWriter(fps=25, codec='mpeg4', bitrate=5000) 
plt.figure() 
fig, ax = plt.subplots(1, figsize=(12.8,7.2), dpi=100)


# Open the stream to the output video, which will be built frame by frame
with writer.saving(fig, output_video_path, dpi=100): 

    frame_number = 0


    # Load each frame of the input video
    for img_name in tqdm(in_image_names):

        frame_number += 1

        # Take next frame of the video and get detections
        pil_img = Image.open(input_path + img_name)

        # Store the image in a plt figure    
        img = np.array(pil_img)
        ax.imshow(img)

        # Extract ground truth boxes of the current frame
        current_boxes = gt_boxes[ gt_boxes[:,0] == frame_number ]


        # Draw over the image the ground truth boxes (if any) 
        if current_boxes is not None:

            # For each object (charachterized by the obj_id), 
            # draw the corresponding bounding box
            for frame_id, obj_id, x1, y1, box_w, box_h in current_boxes:

                color = colors[int(obj_id) % len(colors)]
                color = [i / 255 for i in color]
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)


        # Save the current frame with boxes inside the video
        plt.axis('off')
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        writer.grab_frame() 
        plt.cla()   # clear axis