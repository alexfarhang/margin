import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from util.nero import Nero
from util.data import normalize_data, normalize_data_10_class

from generalization_bounds import bartlett_X_norm, collect_parameters_multiclass

from tqdm import tqdm


class NetScaleInit(nn.Module):
    """Simple MLP class for scale init experiments.  Arbitrarily deep MLP
     with the first layer initialized ~ N(0, sigma), and all other layers
     ~ xavier/glorot"""
    def __init__(self, depth, width, k_classes, sigma):
        super(NetScaleInit, self).__init__()
        self.initial = nn.Linear(784, width, bias=False)
        self.initial.weight.data.normal_(mean=0, std=sigma)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        # do these as xavier init, as well as the final layer
        for i in range(depth-2):
            a = self.layers[i].weight.data
            self.layers[i].weight.data = xavier_uniform_(a)

        self.final = nn.Linear(width, k_classes, bias=False)
        a = self.final.weight.data
        self.final.weight.data = xavier_uniform_(a)


    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)
