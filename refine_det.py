from base_models import *
from modules import *
import torch.nn as nn

class refine_det(nn.Module):
    def __init__(self, backbone, model_conf):
        super(refine_det, self).__init__()
        self.backbone = backbone
        self.ARM = ARM(model_conf['num_bbox'])
        self.TCB = TCB()
        self.ODM = ODM()

    def forward(self, x):

        # we first run the backbone network(VGG16) and extract
        # the layers: conv4_3 and conv5_3
        conv4_3, conv5_3 = self.backbone(x)

        # run through the ARM module
        # input  : feature maps(conv4_3, conv5_3) from the backbone network
        # output : (1) list of featuremaps (l2norm(conv4_3), l2norm(conv5_3), conv6_1 and conv6_2)
        #          (2) list of ARM location
        featmap_list, armloc_list, armconf_list = self.ARM(conv4_3, conv5_3)

        # run through the TCB blocks
        # input  : list of ARM outputs.
        # output : list of TCB outputs.
        TCB_outputs = self.TCB(featmap_list)

        # run through the ODM module
        # input  : list of TCB outputs, list of refined anchors.
        # output : list of ODM outputs. 
        ODM_outputs = self.ODM(TCB_outputs)

        # return ARM_outputs and ODM outputs to comput the loss
        return featmap_list, ODM_outputs


'''
import json
model_conf = json.loads(open('conf.json').read())['model_conf']
backbone = VGG16()
model = refine_det(backbone, model_conf)
print(model)
'''