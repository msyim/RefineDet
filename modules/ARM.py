import torch.nn as nn
from layers import L2NormLayer

class ARM(nn.Module):
    def __init__(self, num_bbox):
        super(ARM, self).__init__()

        # Max pooling layer for conv5_3 input
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        # L2Norm Layers to be applied on conv4_3 and conv5_3
        self.conv4_l2norm = L2NormLayer(512, 8)
        self.conv5_l2norm = L2NormLayer(512, 10)

        # FC6 and FC7 in VGG are converted into conv layers
        self.conv_fc7 = nn.Sequential(
            nn.Conv2d(512,  1024, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU()
        )

        # Extra layers(conv6_1 and conv6_2) added to capture high-level information
        self.conv6_2  = nn.Sequential(
            nn.Conv2d(1024, 256,  kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256,  512,  kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Filters to predict ARM loc and conf. scores
        self.conv4_3_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)
        self.conv5_3_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)
        self.convfc7_loc  = nn.Conv2d(1024, num_bbox * 4, kernel_size=3, padding=1)
        self.conv6_2_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)

        self.conv4_3_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)
        self.conv5_3_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)
        self.convfc7_conf = nn.Conv2d(1024, num_bbox * 2, kernel_size=3, padding=1)
        self.conv6_2_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)

    def forward(self, conv4_3, conv5_3):

        # The following 4 feature maps will be used in training
        # 1. L2Normed conv4_3
        # 2. L2Normed conv5_3
        # 3. conv_fc7
        # 5. conv6_2 (one of the two extra layers to capture high-level information)
        l2_conv4_3 = self.conv4_l2norm(conv4_3)
        l2_conv5_3 = self.conv5_l2norm(conv5_3)
        conv_fc7   = self.conv_fc7(self.mp(conv5_3))
        conv6_2    = self.conv6_2(conv_fc7)

        # Apply ARM loc
        conv4_3_loc = self.conv4_3_loc(l2_conv4_3)
        conv5_3_loc = self.conv5_3_loc(l2_conv5_3)
        convfc7_loc = self.convfc7_loc(conv_fc7)
        conv6_2_loc = self.conv6_2_loc(conv6_2)

        # Apply ARM conf
        conv4_3_conf = self.conv4_3_conf(l2_conv4_3)
        conv5_3_conf = self.conv5_3_conf(l2_conv5_3)
        convfc7_conf = self.conv6_1_conf(conv_fc7)
        conv6_2_conf = self.conv6_2_conf(conv6_2)

        featmap_list = [l2_conv4_3, l2_conv5_3, conv_fc7, conv6_2]
        armloc_list  = [conv4_3_loc, conv5_3_loc, convfc7_loc, conv6_2_loc]
        armconf_list = [conv4_3_conf, conv5_3_conf, convfc7_conf, conv6_2_conf]

        return featmap_list, armloc_list, armconf_list

