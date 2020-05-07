import torch.nn as nn
import torch
from utils import get_anchors, log_sum_exp, match_boxes
import torch.nn.functional as F

def merge_list(tensor_list, num_classes):
    # Permute the tensors in the list so that each tensor has shape (batch, fmap_width, fmap_height, num_channels)
    output = [ t.permute(0,2,3,1).contiguous() for t in tensor_list]
    # Reshape the tensors in the list to have shape: (batch, fmap_width * fmap_height, num_channels)
    output = [ t.view(t.size(0), -1, num_classes) for t in output ]
    # Concat the tensors in the list
    # output tensor shape: (batch, *, num_channels)
    return torch.cat(output, dim=1)

class MultiBoxLoss(nn.Module):
    def __init__(self, conf):
        super(MultiBoxLoss, self).__init__()

        self.num_classes = conf['model_conf']['num_classes']
        self.bbox_conf = conf['bbox_conf']
        self.variance = self.bbox_conf['variance']
        self.use_gpu = 1
        self.negpos_ratio = 0.3
        self.threshold = 0.5
        self.theta = 0.01
        self.anchors = get_anchors(conf['bbox_conf'])


    def forward(self, arm_loc, arm_conf, odm_loc, odm_conf, targets, mode='ARM'):

        # arm_loc  : a list of tensors of shape: (batch, num_bbox * num_channels, fmap_width, fmap_height)
        # arm_conf : a list of tensors of shape: (batch, num_bbox * num_classes,  fmap_width, fmap_height)
        # odm_loc  : a list of tensors of shape analogous to arm_loc
        # odm_conf : a list of tensors of shape analogous to odm_conf

        num_classes = 2 if mode=='ARM' else self.num_classes

        # 1. reshape and merge the tensors in the lists
        arm_loc  = merge_list(arm_loc, 4)
        arm_conf = merge_list(arm_conf, 2)
        odm_loc  = merge_list(odm_loc, 4)
        odm_conf = merge_list(odm_conf, num_classes)

        if mode == 'ARM':
            loc_data, conf_data = arm_loc, arm_conf
        else:
            loc_data, conf_data = odm_loc, odm_conf
        num = loc_data.size(0)

        # 2. Get anchors(prior boxes)
        anchors = self.anchors
        num_priors = (anchors.size(0))

        # 3. Match anchors and GT boxes
        loc_t, conf_t = match_boxes(self.threshold, targets, anchors, self.variance, num_classes, arm_loc, mode)

        loc_t.requires_grad = False
        conf_t.requires_grad = False

        if mode == "ODM":
            P = F.softmax(arm_conf, 2)
            arm_conf_tmp = P[:,:,1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0
        #print(pos.size())
        #num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #print(loss_c.size())

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        #print(num_pos.size(), num_neg.size(), neg.size())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print(pos_idx.size(), neg_idx.size(), conf_p.size(), targets_weighted.size())
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        #N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        #print(N, loss_l, loss_c)
        return loss_l, loss_c




