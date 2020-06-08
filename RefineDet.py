from modules import ARM, ODM, TCB
from layers import Detect_RefineDet
from utils import get_anchors
import torch
import torch.nn as nn
import torch.nn.init as init
import time

class refine_det(nn.Module):
    def __init__(self, backbone, conf):
        super(refine_det, self).__init__()
        self.model_conf = conf['model_conf']
        self.bbox_conf  = conf['bbox_conf']
        self.det_conf   = conf['det_conf']
        self.backbone = backbone
        self.ARM = ARM(self.model_conf['num_bbox'])
        self.TCB = TCB(self.model_conf['tcb_channels'])
        self.ODM = ODM(self.model_conf['num_bbox'], self.model_conf['num_classes'], 4)

        # detection purposes
        self.softmax = nn.Softmax(dim=-1)
        self.det_layer = Detect_RefineDet(self.model_conf['num_classes'], self.bbox_conf['variance'], self.det_conf)

    def weight_initialize(self):
        def xavier(param):
            init.xavier_uniform_(param)

        def init_xavier(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier(m.weight.data)
                m.bias.data.zero_()

        self.ARM.apply(init_xavier)
        self.TCB.apply(init_xavier)
        self.ODM.apply(init_xavier)
        self.backbone.apply(init_xavier)

    def tensorize(self, t_list, num_classes):

        output = [ t.permute(0,2,3,1).contiguous() for t in t_list ]
        output = [ t.view(len(t), -1, num_classes) for t in output ]
        output = torch.cat(output, 1)
        return output

    def forward(self, x, mode='train', anchors=None):

        # we first run the backbone network(VGG16) and extract
        # the feature maps: conv4_3, conv5_3, conv_fc7, conv6_2
        conv4_3, conv5_3, conv_fc7, conv6_2 = self.backbone(x)

        # run through the ARM module
        # input  : feature maps(conv4_3, conv5_3, conv_fc7 and conf6_1) from the backbone network
        # output : (1) list of featuremaps (l2norm(conv4_3), l2norm(conv5_3), conv6_1 and conv6_2)
        #          (2) list of ARM location
        #          (3) list of ARM confidence
        arm_loc, arm_conf = self.ARM(conv4_3, conv5_3, conv_fc7, conv6_2)

        # run through the TCB blocks
        # input  : (1) list of featuremaps (l2norm(conv4_3), l2norm(conv5_3), conv6_1 and conv6_2)
        # output : list of TCB outputs(refer to TCB module for details)
        TCB_outputs = self.TCB([conv4_3, conv5_3, conv_fc7, conv6_2])

        # run through the ODM module
        # input  : list of TCB outputs
        # output : list of odm locs and list of odm conf(class confidence)
        odm_loc, odm_conf = self.ODM(TCB_outputs)

        if mode == 'train':
            # return ARM_outputs and ODM outputs to compute the loss
            return arm_loc, arm_conf, odm_loc, odm_conf
        else:
            # Tensorize the lists
            model_start = time.time()
            arm_loc  = self.tensorize(arm_loc,  4)
            arm_conf = self.tensorize(arm_conf, 2)
            odm_loc  = self.tensorize(odm_loc,  4)
            odm_conf = self.tensorize(odm_conf, self.model_conf['num_classes'])

            arm_conf = self.softmax(arm_conf)
            odm_conf = self.softmax(odm_conf)
            model_taken = time.time() - model_start

            detect_start = time.time()
            anchors.type(type(x.data))
            final_layer = self.det_layer.forward(arm_loc, arm_conf, odm_loc, odm_conf, anchors)
            detect_taken = time.time() - detect_start

            return final_layer 

