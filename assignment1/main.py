import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms as transforms

test_img = Image.open("samples/sampleInput.jpeg")
print(test_img)

max_dimension = 1000

if test_img.size[0] > max_dimension or test_img.size[1] > max_dimension:
    test_img.thumbnail((max_dimension, max_dimension))

transform = transforms.Compose([transforms.ToTensor()])

test_img_tensor = transform(test_img).unsqueeze(dim=0)

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

with torch.no_grad():
    preds = model(test_img_tensor)

COCO_INSTANCE_CATEGORY_NAMES = [
    '', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

img = test_img_tensor.squeeze().detach().cpu().numpy()
img = img.transpose(1, 2, 0)

if img.shape[0] > max_dimension or img.shape[1] > max_dimension:
    img_pil = Image.fromarray((img * 255).astype('uint8'))
    img_pil.thumbnail((max_dimension, max_dimension))
    img = np.array(img_pil) / 255.0

fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(img)

for box, label, score in zip(preds[0]['boxes'].detach().cpu().numpy(),
                             preds[0]['labels'].detach().cpu().numpy(),
                             preds[0]['scores'].detach().cpu().numpy()):
    if score > 0.5:
        x1, y1, x2, y2 = box
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'))
        plt.text(x1, y1, f'{label_name} {score:.2f}', color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')
plt.savefig("output_image.jpeg", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
