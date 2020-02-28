import torch.nn as nn
from layers import L2NormLayer

class ARM(nn.Module):
    def __init__(self, num_bbox):
        super(ARM, self).__init__()

        self.num_bbox = num_bbox
        # L2Norm Layers to be applied on conv4_3 and conv5_3
        self.conv4_l2norm = L2NormLayer(512, 8)
        self.conv5_l2norm = L2NormLayer(512, 10)

        # FC6 and FC7 in VGG are converted into conv layers
        self.conv_fc6 = nn.Sequential(
            nn.Conv2d(512,  1024, kernel_size=3, padding=3, dilation=3),
            nn.ReLU()
        )
        self.conv_fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU()
        )

        # Extra layers(conv6_1 and conv6_2) added to capture high-level information
        self.conv6_1  = nn.Sequential(
            nn.Conv2d(1024, 256,  kernel_size=1),
            nn.ReLU()
        )
        self.conv6_2  = nn.Sequential(
            nn.Conv2d(256,  512,  kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Filters to predict ARM loc and conf. scores
        self.conv4_3_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)
        self.conv4_3_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)

        self.conv5_3_loc  = nn.Conv2d(512, num_bbox * 4, kernel_size=3, padding=1)
        self.conv5_3_conf = nn.Conv2d(512, num_bbox * 2, kernel_size=3, padding=1)

        self.conv6_1_loc  = nn.Conv2d(1024, num_bbox * 4, kernel_size=3, padding=1)
        self.conv6_1_conf = nn.Conv2d(1024, num_bbox * 2, kernel_size=3, padding=1)

        self.conv6_2_loc  = nn.Conv2d(1024, num_bbox * 4, kernel_size=3, padding=1)
        self.conv6_2_conf = nn.Conv2d(1024, num_bbox * 2, kernel_size=3, padding=1)

    def forward(self, conv4_3, conv5_3):

        # Apply up to conv_fc7
        conv_fc7 = self.conv_fc7(self.conv_fc6(conv5_3))



