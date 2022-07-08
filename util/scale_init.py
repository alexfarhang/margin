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


# def compute_alignment(vectors, classes):
#     """General implementation of Extreme Memorization gradient/representation
#     alignment.  Requires vectors as a list of lists style so as 
#     np.array([v1, v2...]) or tensor([v1, v2...]).  Classes also must be 
#     numpy or tensor Extreme Memorization: reported average in class gradient/
#     representation alignment
#     """
#     # Create running sum of \Omega_c (in class alignment)
#     temp_sum = 0
#     for each_class in classes.unique():
#         class_vectors = vectors[classes == each_class]
#         temp_sum += class_specific_alignment(class_vectors)
    
#     # Divide temp_sum by the number of classes
#     temp_sum /= len(classes.unique())

#     return temp_sum


# def class_specific_alignment(vectors):
#     """Calculate within class alignment for the inputted (same-class) vectors.
#     Drawing on O(n) implementation from EM github
#     https://github.com/google-research/google-research/blob/master/extreme_memorization/alignment.py"""
#     mean_norm = vectors.norm(p=2, dim=1).mean()
#     denom = torch.where(torch.greater(torch_norms_mean, 0),
#                         torch_norms_mean,
#                         torch.ones(1))
#     normalized_vectors = [v / denom for v in vectors]
#     n = len(normalized_vectors)

#     sum_norm_square = (normalized_vectors.sum(axis=0)).norm(p=2)**2

#     norm_squares_sum = torch.Tensor([v.norm(p=2)**2 for v in normalized_vectors]).sum()
#     alignment = (sum_norm_square - norm_squares_sum) / (n * (n-1))
#     return alignment