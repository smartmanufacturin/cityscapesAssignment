import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

# Cityscapes class definitions
ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle']
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)

# Color mapping for visualization
colors = [
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
    [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
]
label_colours = dict(zip(range(n_classes), colors))

# Alternative color mapping for decode_segmap
label_colors = np.array([
    (255, 99, 71), (255, 105, 180), (135, 206, 235), (255, 223, 186), (244, 164, 66),
    (255, 215, 0), (255, 69, 0), (255, 255, 0), (34, 139, 34), (152, 251, 152),
    (135, 206, 235), (255, 99, 255), (255, 20, 147), (0, 255, 255), (255, 165, 0),
    (255, 0, 255), (138, 43, 226), (255, 105, 180), (255, 215, 180), (255, 255, 255)
])

def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize a tensor image."""
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def encode_segmap(mask):
    """Encode segmentation mask by mapping valid classes and setting void classes to ignore_index."""
    mask = mask.clone()
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    """Convert grayscale segmentation mask to RGB using Cityscapes color mapping."""
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def decode_segmap_alt(seg):
    """Alternative decode_segmap using different color mapping."""
    r = np.zeros_like(seg).astype(np.uint8)
    g = np.zeros_like(seg).astype(np.uint8)
    b = np.zeros_like(seg).astype(np.uint8)
    for label in range(0, len(label_colors)):
        idx = seg == label
        r[idx] = label_colors[label, 0]
        g[idx] = label_colors[label, 1]
        b[idx] = label_colors[label, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_sample(invimg, encoded_mask, output, sample=7, save_path='result.png'):
    """Visualize input image, ground truth mask, and predicted mask."""
    invimg = inv_normalize(invimg[sample])
    outputx = output.detach().cpu()[sample]
    decoded_mask = decode_segmap(encoded_mask.clone())
    decoded_output = decode_segmap(torch.argmax(outputx, 0))
    
    fig, ax = plt.subplots(ncols=3, figsize=(16, 50), facecolor='white')
    ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2