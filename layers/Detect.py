import torch
from torch.autograd import Function
#from torchvision.ops import nms
import torch.nn as nn
import numpy as np
from config import * 
import time
from utils import decode, center_size, nms, nms_cpu

class Detect_RefineDet(Function):

    num_classes = None
    variance    = None
    det_conf    = None
    conf_threshold = None
    obj_threshold  = None
    nms_threshold  = None
    top_k          = None
    keep_top_k     = None

    def __init__(self, n_classes, var, d_conf):
        Detect_RefineDet.num_classes    = n_classes
        Detect_RefineDet.variance       = var
        Detect_RefineDet.det_conf       = d_conf
        Detect_RefineDet.conf_threshold = d_conf['conf_threshold']
        Detect_RefineDet.obj_threshold  = d_conf['obj_threshold']
        Detect_RefineDet.nms_threshold  = d_conf['nms_threshold']
        Detect_RefineDet.top_k          = d_conf['top_k']
        Detect_RefineDet.keep_top_k     = d_conf['keep_top_k']
        
        if Detect_RefineDet.nms_threshold <= 0: raise ValueError('nms_threshold must be non-negative.')

    @staticmethod
    def forward(arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        """
        Args:
            loc_data: () Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc_data = odm_loc_data.cpu()
        conf_data = odm_conf_data.cpu()
        arm_loc_data = arm_loc_data.cpu()
        arm_conf_data = arm_conf_data.cpu()
        prior_data = prior_data.cpu()

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= Detect_RefineDet.obj_threshold
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, Detect_RefineDet.num_classes, Detect_RefineDet.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, Detect_RefineDet.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        d_total = 0
        n_total = 0
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, Detect_RefineDet.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, Detect_RefineDet.variance)
            #conf_scores = conf_preds[i].clone()
            conf_scores = conf_preds[i]

            for cl in range(1, Detect_RefineDet.num_classes):
                c_mask = conf_scores[cl].gt(Detect_RefineDet.conf_threshold)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #ids, count = nms(boxes, scores, Detect_RefineDet.nms_threshold, Detect_RefineDet.top_k)
                ids_np = nms_cpu(boxes, scores, Detect_RefineDet.nms_threshold, Detect_RefineDet.top_k)
                #ids_np = ids_np[:min(len(ids_np),Detect_RefineDet.top_k)]
                output[i, cl, :len(ids_np)] = torch.cat((scores[ids_np].unsqueeze(1), boxes[ids_np]), 1)

        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < Detect_RefineDet.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output

