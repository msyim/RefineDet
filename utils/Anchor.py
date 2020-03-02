from itertools import product
import torch
from math import sqrt

def get_anchors(bbox_config):

    fm_widths     = bbox_config['fm_widths']
    image_size    = bbox_config['image_size']
    stride_steps  = bbox_config['stride_steps']
    aspect_ratios = bbox_config['aspect_ratios']

    anchors = []
    for stride, fm_width in zip(stride_steps, fm_widths):
        anchor_scale = stride * 4
        for i, j in product(range(fm_width), repeat=2):

            cx = (j + 0.5) / fm_width
            cy = (i + 0.5) / fm_width
            wh = anchor_scale / image_size
            anchors += [ [cx, cy, wh * sqrt(ar), wh / sqrt(ar)] for ar in aspect_ratios ]

    anchors = torch.Tensor(anchors).view(-1, 4)

    return anchors