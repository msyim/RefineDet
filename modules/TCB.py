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

        self.feat_scales = nn.ModuleList([ feature_scale(cs) for cs in channel_sizes ])
        self.pred_layers = nn.ModuleList([ pred_layer() for _ in channel_sizes ])
        self.upsamples   = nn.ModuleList([ nn.ConvTranspose2d(256, 256, 2, 2) for _ in channel_sizes[:1] ])

    def forward(self, featmap_list):

        # apply feature scale layer to all 4 of the feature maps
        scaled     = [ self.feat_scales[i](featmap) for i, featmap in enumerate(featmap_list) ]

        # upsample the last 3 of the feature scaled outputs
        upsampled  = [ self.upsamples[i](s) for i, s in enumerate(scaled[1:]) ]

        # apply element-wise sum to the upsampled with the first 3 of feature scaled
        elwise_sum = [ scaled[i] + upsampled[i] for i in upsampled ]

        # append the last element of scaled to elwise_sum list
        elwise_sum.append(scaled[-1])

        # apply pred layer to the elementwise sum
        TCB_outputs = [ self.pred_layers(i)[s] for i,s in enumerate(elwise_sum) ]

        return TCB_outputs







