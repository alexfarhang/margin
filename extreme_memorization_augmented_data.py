"""Generalized version allowing for Nero, scaledNero, SGD
Vary margins or scale of init
Experiments on network scale and margin (Extreme memorization).
General setup: two layer NN with first layer W ~ N(0, sigma**2), modify
sigma.

"""
#TODO: change the trainer to use generalized_multiclass_train for simplicity
import itertools
import sys
import math
from tqdm import tqdm

import pickle

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd.functional import vhp, hessian
from torchvision import datasets, transforms

sys.path.append("..")

from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from fromage import Fromage
from util.nero import Nero
from util.data import get_data, normalize_data, normalize_data_10_class, get_data_k_class, get_data_augmented_k_classes
from util.trainer import SimpleNet, SimpleNetMultiClass, train_network, train_network_multiclass, train_network_multiclass_combined, train_network_multiclass_scale_label
from util.trainer import train_network_multiclass_scale_label_input_net, generalized_multiclass_train
from util.hessians import *
from util.scale_init import NetScaleInit
from matplotlib import cm
from generalization_bounds import *

# specify the types of experiments here
experiment_flag = 'margin_scale'

opt_string = 'Nero'
# opt_string = 'SGD'
# constraints_string = 'False' # Really only for Nero
# constraints_string = 'False'
constraints_string = 'True'

if constraints_string == 'False':
    constraints_flag = False
elif constraints_string == 'True':
    constraints_flag =  True


if experiment_flag == 'margin_scale':
    param_size = 6
    # param_size = 2
    # scale_vector = np.logspace(-2, 1, num=param_size)
    scale_vector = np.logspace(-1, 1, num=param_size)
    param_vector = scale_vector
    sigma_vector = None
    sigma_base = 0.001
    sigma = sigma_base
    true_data_label_scale = None
    rand_label_scale = None

# elif experiment_flag == 'init_scale':
#     sigma_base = 0.001
#     # sigma_vector = sigma_base
#     sigma = sigma_base
#     sigma_vector = [sigma_base * 2**i for i in range(11)]
#     true_data_label_scale = 1
#     rand_label_scale = 1    
#     param_vector = sigma_vector
#     scale_vector = [1]

num_networks = len(param_vector)

epochs = 4000 #3000
lr  = 0.01
depth = 2
width = 2048
k_classes = 10
num_train_examples = 500 #2000
num_test_examples = 1000
# lr_decay = 1
lr_decay = 0.9995
# tqdm_flag = False
tqdm_flag = False

if opt_string == 'Nero':
    cur_opt = Nero
    optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
    optimizer_kwargs_to_save = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
elif opt_string == 'SGD':
    cur_opt = SGD
    optimizer_kwargs = {'lr':lr}#, 'beta':0.999, 'constraints':True}
    optimizer_kwargs_to_save = {'lr':lr}#, 'beta':0.999, 'constraints':True}


to_spect_norm = False

criterion = nn.MSELoss()
early_stop = False

params_set = {'num_networks': num_networks,
              'epochs': epochs,
              'lr': lr,
              'depth': depth,
              'width': width,
              'num_train_examples': num_train_examples,
              'lr_decay': lr_decay,
              'optimizer': cur_opt,
              'optimizer_kwargs': optimizer_kwargs_to_save,
              'true_data_label_scale': true_data_label_scale,
              'rand_label_scale': rand_label_scale,
              'sigma_vector': sigma_vector,
              'loss_function': criterion,
              'scale_vector': scale_vector,
              'experiment_flag': experiment_flag,
              'k_classes': k_classes
              }


true_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'X_norms': [],
                     'sigma': [],
                     }

rand_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'X_norms': [],
                     'sigma': []
                     }

# for net in tqdm(range(num_networks)):
for net, cur_param in enumerate(tqdm(param_vector)):
    if experiment_flag == 'init_scale':
        sigma = cur_param
        cur_scale = 1
    elif experiment_flag == 'margin_scale':
        cur_scale = cur_param

    true_data_label_scale = cur_scale

    full_batch_train_loader, full_batch_test_loader, train_loader, test_loader = get_data_augmented_k_classes(
        num_train_examples=num_train_examples,
        num_test_examples=num_test_examples, 
        batch_size=500,
        k_classes=k_classes)



    model = NetScaleInit(depth, width, k_classes, sigma)
    train_acc, test_acc, model, init_weights, outputs, correct_class_outputs,\
    other_class_outputs, X_norm, targets = \
        generalized_multiclass_train(model,
                                        criterion,
                                        full_batch_train_loader,
                                        full_batch_test_loader, 
                                        k_classes, 
                                        cur_opt, 
                                        optimizer_kwargs,
                                        epochs,
                                        lr_decay,
                                        to_spect_norm=False,
                                        label_scale=true_data_label_scale,
                                        return_init=True,
                                        return_margins=True,
                                        early_stop=early_stop,
                                        tqdm_flag = tqdm_flag)


    fro_norm = []
    # fro_norm_rand = []
    spectral_norm = []
    # spectral_norm_rand = []
    model.cpu()
    # rand_model.cpu()
    for idx, params in enumerate(model.parameters()):#, rand_model.parameters())):

        # p, p_rand = params
        p = params
        p = p.detach()
        # p_rand = p_rand.detach()
        fro_norm.append(np.linalg.norm(p, ord='fro'))
        # fro_norm_rand.append(np.linalg.norm(p_rand, ord='fro'))

        spectral_norm.append(np.linalg.norm(p, ord=2))
        # spectral_norm_rand.append(np.linalg.norm(p_rand, ord=2))

    spect_complex = bartlett_spectral_complexity(model, ref_M=init_weights)


    true_data_results['train_acc_list'].append(train_acc)
    true_data_results['test_acc_list'].append(test_acc)
    true_data_results['fro_norms'].append(fro_norm)
    true_data_results['spect_norms'].append(spectral_norm)
    true_data_results['correct_class_outputs'].append(correct_class_outputs)
    true_data_results['other_class_outputs'].append(other_class_outputs)
    true_data_results['bartlett_spect_complexity'].append(spect_complex)
    true_data_results['X_norms'].append(X_norm)
    true_data_results['sigma'].append(sigma)





final_vals_dict = {'true_data_results': true_data_results}#,
                #    'rand_data_results': rand_data_results}

final_vals_dict['parameters'] = params_set


# f = open("generalization_extreme_memorization_margin_scale.pkl", "wb")
f = open(f"faster_extreme_memorization_data_augmented_{opt_string}_{constraints_string}_constraints_{epochs}_epochs_{lr_decay}_lrdecay_{depth}_depth.pkl", "wb")
pickle.dump(final_vals_dict, f)
f.close()
