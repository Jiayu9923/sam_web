import onnxruntime
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")

# load image
image = cv2.imread('img/street_down.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# display image
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

# some set-up
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    ax.imshow(img)
    # return img


checkpoint = "model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

print("Loading -")
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

print("Loading --")
mask_generator = SamAutomaticMaskGenerator(sam)

print("Loading ---")
masks = mask_generator.generate(image)



print("Loading ----")
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('foo.png')
# plt.show()




# # Get input names
# input_names = [input.name for input in ort_session.get_inputs()]
# print('Input names: ', input_names)
#
# # Get input shapes
# input_shapes = [input.shape for input in ort_session.get_inputs()]
# print('Input shapes: ', input_shapes)