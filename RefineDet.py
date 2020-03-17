from base_models import VGG16
from modules import ARM, ODM, TCB
import torch.nn as nn
import torch

## test
from utils import SSDAugmentation, VOCDetection
import torch.utils.data as data

class refine_det(nn.Module):
    def __init__(self, backbone, model_conf):
        super(refine_det, self).__init__()
        self.backbone = backbone
        self.ARM = ARM(model_conf['num_bbox'])
        self.TCB = TCB([512,512,1024,512])
        self.ODM = ODM(model_conf['num_bbox'], model_conf['num_classes'], 4)

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

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

import json
model_conf = json.loads(open('conf.json').read())['model_conf']
backbone = VGG16()
model = refine_det(backbone, model_conf)
#print(model)
#dummy = torch.rand([3,3,512,512])
#al, ac, ol, oc = model(dummy)
dataset = VOCDetection(root='/Users/minsub/PycharmProjects/data', transform=SSDAugmentation(512))
dl = data.DataLoader(dataset = dataset, collate_fn=detection_collate, batch_size=5, shuffle=True)
for x, y in dl:
    al, ac, ol, oc = model(x)
    al_temp = torch.cat([t.permute(0,2,3,1).contiguous().view(t.size()[0], -1, 4) for t in al], dim=1)
    print(al_temp.size())
    print("al shape:", al[0].size())
    print("ac shape:", ac[0].size())
    print("ol shape:", ol[0].size())
    print("oc shape:", oc[0].size())
    print(y)
    break

