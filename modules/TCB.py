import torch.nn as nn

def feature_scale(input_channel):
    layer = nn.Sequential(
        nn.Conv2d(input_channel, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1)
    )
    return layer

def pred_layer():
    layer = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU()
    )
    return layer

class TCB(nn.Module):
    def __init__(self, channel_sizes):
        super(TCB, self).__init__()

        # Default Channel sizes : [512,512,1024,512]
        self.feat_scales = nn.ModuleList([ feature_scale(cs) for cs in channel_sizes ])
        self.pred_layers = nn.ModuleList([ pred_layer() for _ in channel_sizes ])
        self.upsamples   = nn.ModuleList([ nn.ConvTranspose2d(256, 256, 2, 2) for _ in channel_sizes[:-1] ])

    def forward(self, featmap_list):

        # apply feature scale layer to all 4 of the feature maps
        scaled = [ self.feat_scales[i](featmap) for i, featmap in enumerate(featmap_list) ]

        # Starting from the last block, upsample and perform elementwise sum
        # with the previous block
        TCB_outputs = []
        for i, b in enumerate(scaled[::-1]):
            if i: TCB_outputs = [ self.upsamples[3-i](TCB_outputs[0]) + b ] + TCB_outputs
            else: TCB_outputs.append(b)

        TCB_outputs = [ self.pred_layers[i](tcb) for i, tcb in enumerate(TCB_outputs) ]

        return TCB_outputs







