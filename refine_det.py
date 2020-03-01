from base_models import *
from modules import *
import torch.nn as nn

class refine_det(nn.Module):
    def __init__(self, backbone, model_conf):
        super(refine_det, self).__init__()
        self.backbone = backbone
        self.ARM = ARM(model_conf['num_bbox'])
        self.TCB = TCB([512,512,1024,512])
        self.ODM = ODM(model_conf['num_bbox'], model_conf['num_classes'], 4)

    def forward(self, x):

        # we first run the backbone network(VGG16) and extract
        # the layers: conv4_3 and conv5_3
        conv4_3, conv5_3 = self.backbone(x)

        # run through the ARM module
        # input  : feature maps(conv4_3, conv5_3) from the backbone network
        # output : (1) list of featuremaps (l2norm(conv4_3), l2norm(conv5_3), conv6_1 and conv6_2)
        #          (2) list of ARM location
        #          (3) list of ARM confidence
        featmaps, arm_loc, arm_conf = self.ARM(conv4_3, conv5_3)

        # run through the TCB blocks
        # input  : (1) list of featuremaps (l2norm(conv4_3), l2norm(conv5_3), conv6_1 and conv6_2)
        # output : list of TCB outputs(refer to TCB module for details)
        TCB_outputs = self.TCB(featmaps)

        # run through the ODM module
        # input  : list of TCB outputs
        # output : list of odm locs and list of odm conf(class confidence)
        odm_loc, odm_conf = self.ODM(TCB_outputs)

        # return ARM_outputs and ODM outputs to comput the loss
        return arm_loc, arm_conf, odm_loc, odm_conf

import json
model_conf = json.loads(open('conf.json').read())['model_conf']
backbone = VGG16()
model = refine_det(backbone, model_conf)
print(model)
