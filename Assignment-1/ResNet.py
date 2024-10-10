import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
from pathlib import Path
import random
import os
from skimage import io
from pycocotools.coco import COCO
import matplotlib.patches as patches
import time
import mxnet as mx

DATA_PATH = "/kaggle/input/coco-2017-dataset/coco2017/val2017/"
annotation_file = Path('/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json')

coco_api = COCO(annotation_file)
image_ids = coco_api.getImgIds()

def get_random_image():
    img_id = random.choice(image_ids)
    img_metadata = coco_api.loadImgs([img_id])
    img = io.imread(DATA_PATH + img_metadata[0]['file_name'])
    ann_ids = coco_api.getAnnIds(imgIds=[img_id])
    annotations = coco_api.loadAnns(ann_ids)
    return img, annotations

def test_model(model, image):
    mx_image = mx.nd.array(image)
    x, original_image = data.transforms.presets.rcnn.transform_test(mx_image)

    start_time = time.time()
    box_ids, scores, bboxes = model(x)
    elapsed_time = time.time() - start_time

    ax = utils.viz.plot_bbox(original_image, bboxes[0], scores[0], box_ids[0], class_names=model.classes)
    return ax, elapsed_time

rcnn_model = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)

def display_ground_truth(image, boxes):
    copy_image = image.copy()
    fig, ax = plt.subplots()
    ax.imshow(copy_image)
    for box in boxes:
        rect = patches.Rectangle(
            (int(box['bbox'][0]), int(box['bbox'][1])),
            int(box['bbox'][2]), int(box['bbox'][3]),
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title("Ground Truth")
    plt.show()

random_image, random_annotations = get_random_image()
plt.imshow(random_image)
display_ground_truth(random_image, random_annotations)

ax, elapsed_time = test_model(rcnn_model, random_image)
ax.set_title(f"Faster R-CNN \n Time taken: {round(elapsed_time, 4)} seconds")
plt.show()

random_image, random_annotations = get_random_image()
display_ground_truth(random_image, random_annotations)

ax1, elapsed_time_1 = test_model(rcnn_model, random_image)
ax1.set_title(f"Faster R-CNN \n Time taken: {round(elapsed_time_1, 4)} seconds")
plt.show()

def generate_random_voc_image():
    voc_image_path = "/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/"
    img_path = random.choice(os.listdir(voc_image_path))
    return io.imread(voc_image_path + img_path)

voc_image = generate_random_voc_image()
ax2, elapsed_time_2 = test_model(rcnn_model, voc_image)
ax2.set_title(f"Faster R-CNN on VOC Image \n Time taken: {round(elapsed_time_2, 4)} seconds")
plt.show()

custom_image_paths = [
    "/kaggle/input/test-model-by-me/OIP.jpg",
    "/kaggle/input/test-model-by-me/test4.jpg",
    "/kaggle/input/test-model-by-me/test5.jpg",
    "/kaggle/input/test-model-by-me/test8.jpg",
    "/kaggle/input/test-model-by-me/test 9.PNG"
]

for path in custom_image_paths:
    img = cv2.imread(path)
    ax, elapsed_time = test_model(rcnn_model, img)
    ax.set_title(f"Faster R-CNN \n Time taken: {round(elapsed_time, 4)} seconds")
    plt.show()
