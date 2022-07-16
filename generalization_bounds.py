"""Generalization functions"""
import numpy as np
import torch
from util.data import normalize_data, normalize_data_10_class

def bartlett_spectral_complexity(model, ref_M=None):
    """The spectral complexity from 'Spectrally normalized margin bounds' 
    Bartlett et al
    Uses reference matrix of 0 by default.
    """

    model_params = [x.detach().cpu().numpy() for x in model.parameters()]
    
    # Lipschitz constant of the nonlinearity for the layer.  1 for all
    #  experiments
    temp_rho = [1 for x in range(len(model_params))]

    spect_prod = 1
    weight_sum = 0
    for idx, (layer, rho_i) in enumerate(zip(model_params, temp_rho)):
        # product of spectral norms
        sigma_i = np.linalg.norm(layer, ord=2)
        spect_prod *= sigma_i * rho_i

        # sum 
        if ref_M == None:
            ref_mat = np.zeros(layer.shape)
        else:
            ref_mat = ref_M[idx]
        # Checked. correct (10/21/21)
        temp_normalized_weight = np.linalg.norm(layer.T - ref_mat.T, ord=2, axis=0)
        temp_normalized_weight = np.linalg.norm(temp_normalized_weight, ord=1)
        temp_normalized_weight = temp_normalized_weight**(2/3)
        temp_normalized_weight /= sigma_i ** (2/3)
        weight_sum += temp_normalized_weight

    R_A = spect_prod * weight_sum**(3/2)
    return R_A

def bartlett_X_norm(X):
    """Calculates the norm of the data according to Bartlett 1.1"""
    cur_norm = 0
    for i in X:
        cur_norm += np.linalg.norm(i)**2
    return np.sqrt(cur_norm)


def collect_parameters_multiclass(model, data_loader):
    """get parameters to feed into the other functions.  10 class
    returns: 
        all_x: features of train_data
        all_y: labels of train_data
        W: maximum width of model
        outputs: model outputs    

    """
    W = max([p.shape[1] for p in model.parameters()])
    
    for idx, (x, y) in enumerate(data_loader):
        if idx == 0:
            all_x, all_y = normalize_data_10_class(x, y)
        else:
            x, y = normalize_data_10_class(x, y)
            all_x = torch.cat((all_x, x), dim=0)
            all_y = torch.cat((all_y, y), dim=0)
   
    
    return all_x, all_y, W
