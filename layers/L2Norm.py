import torch
import torch.nn as nn
import torch.nn.init as init

class L2NormLayer(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2NormLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        init.constant_(self.weight, scale)
        self.eps = 1e-10

    def forward(self, feature_map):
        #norm   = feature_map.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        norm   = feature_map.pow(2).sum(dim=1, keepdim=True)
        norm   = torch.sqrt(norm + self.eps)
        normed = torch.div(feature_map, norm)
        output = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(normed) * normed
        return output
