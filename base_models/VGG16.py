import torch.nn as tnn
from layers import L2NormLayer

def conv_layer(c_in, c_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(c_in, c_out, kernel_size=k_size, padding=p_size),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(c_in, c_out, k_list, p_list):

    layers = []
    num_layers = len(c_in)

    for i in range(num_layers):
        conv = conv_layer(c_in[i], c_out[i], k_list[i], p_list[i])
        layers.append(conv)

    return tnn.Sequential(*layers)

class VGG16(tnn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # Conv blocks 
        self.conv1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1])
        self.conv2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1])
        self.conv3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1])
        self.conv4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1])
        self.conv5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1])

        self.mp    = tnn.MaxPool2d(kernel_size=2, stride=2)

        # L2Norm Layers to be applied on conv4_3 and conv5_3
        self.conv4_l2norm = L2NormLayer(512, 8)
        self.conv5_l2norm = L2NormLayer(512, 10)

        # FC6 and FC7 in VGG are converted into conv layers
        self.conv_fc7 = tnn.Sequential(
            tnn.Conv2d(512,  1024, kernel_size=3, padding=3, dilation=3),
            tnn.ReLU(),
            tnn.Conv2d(1024, 1024, kernel_size=1),
            tnn.ReLU()
        )

        # Extra layers(conv6_1 and conv6_2) added to capture high-level information
        self.conv6_2  = tnn.Sequential(
            tnn.Conv2d(1024, 256,  kernel_size=1),
            tnn.ReLU(),
            tnn.Conv2d(256,  512,  kernel_size=3, stride=2, padding=1),
            tnn.ReLU()
        )

    def forward(self, x):
        out      = self.mp(self.conv1(x))
        out      = self.mp(self.conv2(out))
        out      = self.mp(self.conv3(out))
        conv4_3  = self.conv4(out)
        conv5_3  = self.conv5(self.mp(conv4_3))
        conv_fc7 = self.conv_fc7(self.mp(conv5_3))
        conv6_2  = self.conv6_2(conv_fc7)

        return self.conv4_l2norm(conv4_3), self.conv5_l2norm(conv5_3), conv_fc7, conv6_2
