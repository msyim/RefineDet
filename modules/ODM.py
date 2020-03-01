import torch.nn as nn

class ODM(nn.Module):
    def __init__(self, num_bbox, num_classes, TCM_outputlen):
        super(ODM, self).__init__()

        self.num_bbox = num_bbox
        self.TCM_outputlen = TCM_outputlen
        self.odm_loc  = nn.ModuleList([nn.Conv2d(256, num_bbox*4, kernel_size=3, padding=1) for _ in range(TCM_outputlen)] )
        self.odm_conf = nn.ModuleList([nn.Conv2d(256, num_bbox*num_classes, kernel_size=3, padding=1) for _ in range(TCM_outputlen)])

    def forward(self, TCB_outputs):

        odm_loc  = [ self.odm_loc[i](TCB_outputs[i]) for i in range(self.TCM_outputlen) ]
        odm_conf = [ self.odm_conf[i](TCB_outputs[i]) for i in range(self.TCM_outputlen) ]

        return odm_loc, odm_conf


