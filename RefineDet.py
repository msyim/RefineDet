from modules import ARM, ODM, TCB
import torch.nn as nn
import torch.nn.init as init

class refine_det(nn.Module):
    def __init__(self, backbone, model_conf):
        super(refine_det, self).__init__()
        self.backbone = backbone
        self.ARM = ARM(model_conf['num_bbox'])
        self.TCB = TCB([512,512,1024,512])
        self.ODM = ODM(model_conf['num_bbox'], model_conf['num_classes'], 4)

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

    def forward(self, x):

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

        # return ARM_outputs and ODM outputs to comput the loss
        return arm_loc, arm_conf, odm_loc, odm_conf

