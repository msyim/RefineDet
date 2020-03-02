import torch.nn as nn
from layers import L2NormLayer

class ARM(nn.Module):
    def __init__(self, num_bbox):
        super(ARM, self).__init__()

        # Filters to predict ARM loc and conf. scores
        self.conv4_3_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)
        self.conv5_3_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)
        self.convfc7_loc  = nn.Conv2d(1024, num_bbox * 4, kernel_size=3, padding=1)
        self.conv6_2_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)

        self.conv4_3_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)
        self.conv5_3_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)
        self.convfc7_conf = nn.Conv2d(1024, num_bbox * 2, kernel_size=3, padding=1)
        self.conv6_2_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)

    def forward(self, conv4_3, conv5_3, conv_fc7, conv6_2):

        # Apply ARM loc
        conv4_3_loc = self.conv4_3_loc(conv4_3)
        conv5_3_loc = self.conv5_3_loc(conv5_3)
        convfc7_loc = self.convfc7_loc(conv_fc7)
        conv6_2_loc = self.conv6_2_loc(conv6_2)

        # Apply ARM conf
        conv4_3_conf = self.conv4_3_conf(conv4_3)
        conv5_3_conf = self.conv5_3_conf(conv5_3)
        convfc7_conf = self.convfc7_conf(conv_fc7)
        conv6_2_conf = self.conv6_2_conf(conv6_2)

        armloc_list  = [conv4_3_loc, conv5_3_loc, convfc7_loc, conv6_2_loc]
        armconf_list = [conv4_3_conf, conv5_3_conf, convfc7_conf, conv6_2_conf]

        return armloc_list, armconf_list

