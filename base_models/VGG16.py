import torch
import torch.nn as tnn

def conv_layer(c_in, c_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(c_in, c_out, kernel_size=k_size, padding=p_size)),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(c_in, c_out, k_list, p_list):

    layers = []
    num_layers = len(c_in)

    for i in range(num_layers):
        conv = conv_layer(c_in[i], c_out[i], k_list[i], p_list[i])
        layers.append(conv)
    
    layers.append( tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s) )

    return tnn.Sequential(*layers)

class VGG16(tnn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # Conv blocks 
        self.conv1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.conv2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.conv3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.conv4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.conv5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

    def forward(self, x):
        out     = self.conv1(x)
        out     = self.conv2(out)
        out     = self.conv3(out)
        conv4_3 = self.conv4(out)
        conv5_3 = self.conv5(out)

        return conv4_3, conv5_3 
