import torch.nn as nn
import torch
from utils import *

def merge_list(tensor_list, num_classes):
    # Permute the tensors in the list so that each tensor has shape (batch, fmap_width, fmap_height, num_channels)
    output = [ t.permute(0,2,3,1).contiguous() for t in tensor_list]
    # Reshape the tensors in the list to have shape: (batch, fmap_width * fmap_height, num_channels)
    output = [ t.view(t.size(0), -1, num_classes) for t in output ]
    # Concat the tensors in the list
    # output tensor shape: (batch, *, num_channels)
    return torch.cat(output, dim=1)

class MultiBoxLoss(nn.Module):
    def __init__(self, arm_loc, arm_conf, odm_loc, odm_conf, target):
        super.__init__(self, MultiBoxLoss)

        # arm_loc  : a list of tensors of shape: (batch, num_bbox * num_channels, fmap_width, fmap_height)
        # arm_conf : a list of tensors of shape: (batch, num_bbox * num_classes,  fmap_width, fmap_height)
        # odm_loc  : a list of tensors of shape analogous to arm_loc
        # odm_conf : a list of tensors of shape analogous to odm_conf

        # 1. reshape and merge the tensors in the lists
        arm_loc  = merge_list(arm_loc, 4)
        arm_conf = merge_list(arm_conf, 2)
        odm_loc  = merge_list(odm_loc, 4)
        odm_conf = merge_list(odm_conf, 2)

        # 2. Get anchors(prior boxes)
        anchors = get_anchors()


