from base_models import *
from modules import *
import torch.nn as nn

class refine_det(nn.Module):
    def __init__(self, backbone):
        super(refine_det, nn.Module).__init__()
        self.backbone = backbone
        self.ARM = ARM()
        self.TCB = TCB()
        self.ODM = ODM()

    def forward(self, x):

        # we first run the backbone network(VGG16) and extract
        # the layers: conv4_3 and conv5_3
        conv4_3, conv5_3 = self.backbone(x)

        # run through the ARM module
        # input  : feature map(conv5_3) from the backbone network
        # output : (1) list of ARM outputs
        #          (2) list of refined anchors
        ARM_outputs, refined_anchors = self.ARM(conv4_3, conv5_3)

        # run through the TCB blocks
        # input  : list of ARM outputs.
        # output : list of TCB outputs.
        TCB_outputs = self.TCB(ARM_outputs)

        # run through the ODM module
        # input  : list of TCB outputs, list of refined anchors.
        # output : list of ODM outputs. 
        ODM_outputs = self.ODM(TCB_outputs, refined_anchors)

        # return ARM_outputs and ODM outputs to comput the loss
        return ARM_outputs, ODM_outputs