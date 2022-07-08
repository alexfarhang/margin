"""Generalization bound functions"""
import numpy as np
import torch
from util.data import normalize_data, normalize_data_10_class

def bartlett_spectral_complexity(model, ref_M=None):
    """The spectral complexity from 'Spectrally normalized margin bounds' 
    Bartlett et al
    Uses reference matrix of 0 by default.
    """

    model_params = [x.detach().cpu().numpy() for x in model.parameters()]
    
    # Lipschitz constant of the nonlinearity for the layer
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


def bartlett_1_1_bound(model_, data_loader_, ref_M, correct_class_outputs, other_class_outputs, gamma, delta):
    """Probability of returning the wrong class <= rhs.
    X = training data
     """
    
    # n = X.shape[0]
    # R_gamma = 0

    
    # for x, y, fx in zip(X, Y, outputs):
    #     fx_noty = fx[y] 
    #     if  max(fx) <= (gamma + max())
    # # W is maximum width of the network?
    # rhs = R_gamma + bartlett_X_norm * bartlett_spectral_complexity(
    #     model) / (gamma * n) * np.log(W) + np.sqrt(np.log(1/delta)/n)
    # pass 
    # return rhs
    # X, Y, W, outputs, margins = collect_parameters(model_, data_loader)
    X, Y, W = collect_parameters_multiclass(model_, data_loader_)
    n = X.shape[0]
    R_gamma = 0

    # for x, y, fx in zip(X.detach(), Y.detach(), outputs.detach()):
    #     R_gamma += (fx * y <= gamma)
    # R_gamma = torch.true_divide(R_gamma, n)
    for fx, other_class_out in zip(correct_class_outputs.detach(), other_class_outputs.detach()):
        R_gamma += (fx <= gamma + other_class_out)
    R_gamma = torch.true_divide(R_gamma, n)

    # reshaped_X = X.reshape((X.shape[0], X.shape[2]* X.shape[3]))
    reshaped_X = X
    rhs = R_gamma + (bartlett_X_norm(reshaped_X) * bartlett_spectral_complexity(
        model_, ref_M) / (gamma * n)) * np.log(W) + np.sqrt(np.log(1/delta)/n)
    
    return rhs.item()


def bartlett_1_1_bound_binarized(model_, data_loader, ref_M, gamma, delta):
    """Converted the bartlett 1-1 bound to a binary output analog"""

    X, Y, W, outputs, margins = collect_parameters(model_, data_loader)
    n = X.shape[0]
    R_gamma = 0

    # for x, y, fx in zip(X.detach(), Y.detach(), outputs.detach()):
    #     R_gamma += (fx * y <= gamma)
    # R_gamma = torch.true_divide(R_gamma, n)
    for x, y, fx in zip(X.detach(), Y.detach(), outputs.detach()):
        R_gamma += (2 * fx * y <= gamma)
    R_gamma = torch.true_divide(R_gamma, n)

    reshaped_X = X.reshape((X.shape[0], X.shape[2]* X.shape[3]))
    rhs = R_gamma + (bartlett_X_norm(reshaped_X) * bartlett_spectral_complexity(
        model_, ref_M) / (gamma * n)) * np.log(W) + np.sqrt(np.log(1/delta)/n)
    
    return rhs.numpy()[0]


def collect_parameters(model, data_loader):
    """get parameters to feed into the other functions.
    returns: 
        all_x: features of train_data
        all_y: labels of train_data
        W: maximum width of model
        outputs: model outputs    
    """
    W = max([p.shape[1] for p in model.parameters()])
    
    for idx, (x, y) in enumerate(data_loader):
        if idx == 0:
            all_x = x
            all_y = y
        else:
            all_x = torch.cat((all_x, x), dim=0)
            all_y = torch.cat((all_y, y), dim=0)
    
    outputs = model(all_x.reshape((all_x.shape[0], all_x.shape[2]
                                   * all_x.shape[3])))
    reshaped_y = all_y.reshape([all_y.shape[0], 1])
    
    margins = torch.mul(outputs, reshaped_y%2*2-1)
    
    return all_x, all_y, W, outputs, margins.detach().cpu().numpy()


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
            # all_x = x
            # all_y = y
            all_x, all_y = normalize_data_10_class(x, y)
        else:
            x, y = normalize_data_10_class(x, y)
            all_x = torch.cat((all_x, x), dim=0)
            all_y = torch.cat((all_y, y), dim=0)
    
    # outputs = model(all_x.reshape((all_x.shape[0], all_x.shape[2]
    #                                * all_x.shape[3])))
    # reshaped_y = all_y.reshape([all_y.shape[0], 1])
    
    # margins = torch.mul(outputs, reshaped_y%2*2-1)
    
    return all_x, all_y, W#, outputs, margins.detach().cpu().numpy()
        

def spectral_normalization(model, R_A, ref_M, X, outputs):
    """Compute a modified spectral normalization of the outputs"""
    pass


def bartlett_a_5_bound(model_, data_loader_, ref_M, correct_class_outputs, other_class_outputs, gamma, delta):
    """Computes the probability of misclassification for unseen data for a given 
    model, dataset, reference matrix, gamma, delta
    Prioritizes ease of code over compute speed.  Could be sped up
    """
    prod_of_lipschitz = 1
    X, Y, W = collect_parameters_multiclass(model_, data_loader_)
    n = X.shape[0]
    R_gamma = 0
    X_norm = bartlett_X_norm(X)

    for fx, other_class_out in zip(correct_class_outputs.detach(), other_class_outputs.detach()):
        R_gamma += (fx <= gamma + other_class_out)
    R_gamma = torch.true_divide(R_gamma, n)
    R_gamma += 8/n

    # Compute third additive term.  Lipschitz nonlinearities = 1
    model_params = [x.detach().cpu().numpy() for x in model_.parameters()]
    temp_sum = 0
    L = len(model_params)
    for idx, layer in enumerate(model_params):
        if ref_M == None:
            ref_mat = np.zeros(layer.shape)
        else:
            ref_mat = ref_M[idx]
        temp_prod = 1/(L) + np.linalg.norm(np.linalg.norm(layer.T - ref_mat.T, ord=2, axis=0), ord=1)
        for j_idx, layer_j in enumerate(model_params):
            if idx == j_idx:
                continue
            else:
                temp_prod *= (1/(L) + np.linalg.norm(layer_j, ord=2))
        temp_prod = temp_prod**(2/3)
        temp_sum += temp_prod
    
    temp_sum = temp_sum**(3/2)
    third_term = temp_sum*(1 + X_norm)*prod_of_lipschitz*(144*np.log(n)*np.log(2*W))/(gamma*n)
    R_gamma += third_term

    # Compute the fourth additive term.
    fourth_term = 0
    temp_sum_21 = 0
    temp_sum_spect = 0
    for idx, layer in enumerate(model_params):
        if ref_M == None:
            ref_mat = np.zeros(layer.shape)
        else:
            ref_mat = ref_M[idx]
        temp_sum_21 += np.log(2 + L * np.linalg.norm(np.linalg.norm(layer.T - ref_mat.T, ord=2, axis=0), ord=1))
        temp_sum_spect += np.log(2 + L * np.linalg.norm(layer, ord=2))

    second_sqrt = np.log(1/delta) + np.log(2*n/gamma) + 2*np.log(2 + X_norm) + 2*temp_sum_21 + 2*temp_sum_spect
    fourth_term = np.sqrt(9/(2*n))*np.sqrt(second_sqrt)
    R_gamma += fourth_term

    return R_gamma.item()